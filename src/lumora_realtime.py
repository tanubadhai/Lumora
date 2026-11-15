import cv2

import time

import subprocess

from collections import defaultdict, deque

from ultralytics import YOLO





# ============================================================

# LUMORA

# Context-Aware Edge AI Smart Glasses Prototype

#

# General YOLO11s

# +

# Road-specific YOLO11s

# +

# ByteTrack

# +

# Spatial Analysis

# +

# CADE

# +

# Windows SAPI TTS

# ============================================================





# ============================================================

# MODEL PATHS

# ============================================================



GENERAL_MODEL_PATH = "yolo11s.pt"



ROAD_MODEL_PATH = (

    r"C:\Users\Tanu\runs\detect\runs\lumora"

    r"\yolo11s_road_finetune\weights\best.pt"

)





# ============================================================

# MODEL SETTINGS

# ============================================================



GENERAL_CONFIDENCE = 0.35



# Road model is intentionally stricter because it is being

# used on live scenes outside its training distribution.

ROAD_CONFIDENCE = 0.55



IMG_SIZE = 640





# ============================================================

# ROAD MODEL CLASS MAPPING

# ============================================================



ROAD_CLASS_MAPPING = {

    "bicycles": "bicycle",

    "buses": "bus",

    "crosswalks": "crosswalk",

    "motorcycles": "motorcycle",

    "traffic lights": "traffic light",

    "vehicles": "car",

}



# fire hydrants intentionally excluded from live Lumora





# ============================================================

# MOVEMENT SETTINGS

# ============================================================



MOVEMENT_HISTORY = 8

MOVEMENT_THRESHOLD = 0.10





# ============================================================

# CADE SETTINGS

# ============================================================



HIGH_RISK_THRESHOLD = 5

MEDIUM_RISK_THRESHOLD = 3





# ============================================================

# ALERT SETTINGS

# ============================================================



APPROACHING_ALERT_ONLY = True



ALERT_COOLDOWN = 5.0

GLOBAL_ALERT_COOLDOWN = 2.5





# ============================================================

# ROAD MODEL EXECUTION

# ============================================================



# Running two YOLO models on CPU is expensive.

# The road model therefore runs every few frames.

ROAD_MODEL_INTERVAL = 3





# ============================================================

# DUPLICATE DETECTION

# ============================================================



IOU_DUPLICATE_THRESHOLD = 0.40





# ============================================================

# OBJECT COLORS

# ============================================================



OBJECT_COLORS = {



    "person": (255, 0, 255),



    "bicycle": (255, 255, 0),

    "car": (0, 255, 0),

    "motorcycle": (0, 165, 255),

    "bus": (255, 0, 0),

    "truck": (0, 0, 255),

    "train": (128, 0, 128),



    "traffic light": (0, 255, 255),

    "crosswalk": (255, 128, 0),



    "bottle": (0, 255, 255),

    "cup": (255, 128, 0),

    "cell phone": (0, 128, 255),

    "laptop": (128, 255, 0),

    "mouse": (128, 128, 255),



    "backpack": (0, 255, 128),

    "handbag": (255, 128, 128),

    "suitcase": (128, 255, 255),



    "chair": (255, 128, 255),

    "bench": (128, 128, 0),

    "dining table": (0, 128, 128),



    "dog": (0, 165, 255),

    "cat": (255, 165, 0),

    "bird": (255, 255, 255),



    "scissors": (0, 0, 255),

    "knife": (0, 0, 255),



    "default": (0, 255, 0)

}





# ============================================================

# HIGH PRIORITY OBJECTS

# ============================================================



HIGH_PRIORITY_OBJECTS = {



    "person",

    "bicycle",

    "car",

    "motorcycle",

    "bus",

    "truck",

    "train",



    "dog",

    "cat",



    "chair",

    "bench",



    "scissors",

    "knife"

}





# ============================================================

# LOAD MODELS

# ============================================================



print()

print("=" * 60)

print("LUMORA - INITIALIZING")

print("=" * 60)



print()

print("[1/2] Loading general YOLO11s...")



general_model = YOLO(GENERAL_MODEL_PATH)



print("General model loaded.")



print()

print("[2/2] Loading road-specific YOLO11s...")



road_model = YOLO(ROAD_MODEL_PATH)



print("Road model loaded.")



print()

print("Road classes enabled:")



for class_name in ROAD_CLASS_MAPPING.values():

    print("  +", class_name)



print("  - fire hydrant")



print()

print("=" * 60)





# ============================================================

# TEXT TO SPEECH

# ============================================================



def speak_alert(message):



    print(f"[VOICE ALERT] {message}")



    try:



        safe_message = message.replace("'", "''")



        ps_script = (



            "Add-Type -AssemblyName System.Speech; "



            "$speak = New-Object "

            "System.Speech.Synthesis.SpeechSynthesizer; "



            "$speak.Rate = 0; "



            "$speak.Volume = 100; "



            f"$speak.Speak('{safe_message}'); "



            "$speak.Dispose();"

        )



        subprocess.Popen(



            [

                "powershell",

                "-NoProfile",

                "-ExecutionPolicy",

                "Bypass",

                "-Command",

                ps_script

            ],



            stdout=subprocess.DEVNULL,

            stderr=subprocess.DEVNULL

        )



    except Exception as e:



        print(f"[TTS ERROR] {e}")





print("TTS ready.")





# ============================================================

# CAMERA

# ============================================================



cap = cv2.VideoCapture(0)



if not cap.isOpened():



    print("ERROR: Could not open camera.")



    exit()





# ============================================================

# TRACK HISTORY

# ============================================================



track_area_history = defaultdict(

    lambda: deque(maxlen=MOVEMENT_HISTORY)

)



track_position_history = defaultdict(

    lambda: deque(maxlen=MOVEMENT_HISTORY)

)



track_seen_time = {}



last_alert_time = {}



last_alert_message = {}



last_global_alert_time = 0.0





# ============================================================

# ROAD TRACKING

# ============================================================



next_road_track_id = 1000



road_tracks = {}





# ============================================================

# POSITION

# ============================================================



def get_position(x_center, frame_width):



    left_boundary = frame_width / 3

    right_boundary = (frame_width / 3) * 2



    if x_center < left_boundary:



        return "LEFT"



    elif x_center > right_boundary:



        return "RIGHT"



    else:



        return "CENTER"





# ============================================================

# VISUAL PROXIMITY

# ============================================================



def get_visual_proximity(box_area, frame_area):



    area_ratio = box_area / frame_area



    if area_ratio < 0.05:



        return "FAR"



    elif area_ratio < 0.15:



        return "MEDIUM"



    else:



        return "NEAR"





# ============================================================

# IOU

# ============================================================



def calculate_iou(box_a, box_b):



    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b



    intersection_x1 = max(ax1, bx1)

    intersection_y1 = max(ay1, by1)



    intersection_x2 = min(ax2, bx2)

    intersection_y2 = min(ay2, by2)



    intersection_width = max(

        0,

        intersection_x2 - intersection_x1

    )



    intersection_height = max(

        0,

        intersection_y2 - intersection_y1

    )



    intersection_area = (

        intersection_width *

        intersection_height

    )



    area_a = (

        max(0, ax2 - ax1) *

        max(0, ay2 - ay1)

    )



    area_b = (

        max(0, bx2 - bx1) *

        max(0, by2 - by1)

    )



    union_area = area_a + area_b - intersection_area



    if union_area <= 0:



        return 0.0



    return intersection_area / union_area





# ============================================================

# ROAD TRACK ID

# ============================================================



def assign_road_track_id(object_name, box):



    global next_road_track_id



    best_id = None

    best_iou = 0.0



    for track_id, previous in road_tracks.items():



        if previous["object"] != object_name:



            continue



        iou = calculate_iou(

            box,

            previous["box"]

        )



        if iou > best_iou:



            best_iou = iou

            best_id = track_id



    if best_id is not None and best_iou > 0.20:



        road_tracks[best_id]["box"] = box

        road_tracks[best_id]["time"] = time.time()



        return best_id



    new_id = next_road_track_id



    next_road_track_id += 1



    road_tracks[new_id] = {

        "object": object_name,

        "box": box,

        "time": time.time()

    }



    return new_id





# ============================================================

# CLEAN OLD ROAD TRACKS

# ============================================================



def cleanup_road_tracks():



    current_time = time.time()



    expired = []



    for track_id, data in road_tracks.items():



        if current_time - data["time"] > 2.0:



            expired.append(track_id)



    for track_id in expired:



        del road_tracks[track_id]





# ============================================================

# MOVEMENT

# ============================================================



def calculate_movement(track_id):



    history = track_area_history[track_id]



    if len(history) < 6:



        return "STATIONARY"



    values = list(history)



    midpoint = len(values) // 2



    old_values = values[:midpoint]

    recent_values = values[midpoint:]



    old_average = (

        sum(old_values) /

        len(old_values)

    )



    recent_average = (

        sum(recent_values) /

        len(recent_values)

    )



    if old_average == 0:



        return "STATIONARY"



    change_ratio = (

        recent_average -

        old_average

    ) / old_average



    increasing_steps = 0

    decreasing_steps = 0



    for i in range(1, len(values)):



        previous = values[i - 1]

        current = values[i]



        if previous == 0:



            continue



        step_change = (

            current - previous

        ) / previous



        if step_change > 0.015:



            increasing_steps += 1



        elif step_change < -0.015:



            decreasing_steps += 1



    if (

        change_ratio > MOVEMENT_THRESHOLD

        and increasing_steps >= 3

    ):



        return "APPROACHING"



    elif (

        change_ratio < -MOVEMENT_THRESHOLD

        and decreasing_steps >= 3

    ):



        return "MOVING AWAY"



    else:



        return "STATIONARY"





# ============================================================

# CADE

# ============================================================



def calculate_cade(

    position,

    proximity,

    movement,

    object_name

):



    position_score = {



        "CENTER": 2,

        "LEFT": 1,

        "RIGHT": 1

    }



    proximity_score = {



        "NEAR": 2,

        "MEDIUM": 1,

        "FAR": 0

    }



    movement_score = {



        "APPROACHING": 2,

        "STATIONARY": 0,

        "MOVING AWAY": 0

    }



    score = (



        position_score[position]

        +

        proximity_score[proximity]

        +

        movement_score[movement]

    )



    if object_name in HIGH_PRIORITY_OBJECTS:



        if proximity == "NEAR":



            score += 1



    if score >= HIGH_RISK_THRESHOLD:



        risk = "HIGH"



    elif score >= MEDIUM_RISK_THRESHOLD:



        risk = "MEDIUM"



    else:



        risk = "LOW"



    return score, risk





# ============================================================

# ALERT ELIGIBILITY

# ============================================================



def is_alert_candidate(

    object_name,

    position,

    proximity,

    movement,

    score

):



    if proximity == "FAR":



        return False



    if APPROACHING_ALERT_ONLY:



        if movement != "APPROACHING":



            return False



    if score < HIGH_RISK_THRESHOLD:



        return False



    return True





# ============================================================

# ALERT MESSAGE

# ============================================================



def generate_alert(

    object_name,

    position,

    movement

):



    object_name = object_name.replace("\\_", " ")



    if position == "LEFT":



        direction = "from your left"



    elif position == "RIGHT":



        direction = "from your right"



    else:



        direction = "ahead"



    if movement == "APPROACHING":



        return (

            f"{object_name} approaching "

            f"{direction}"

        )



    return (

        f"{object_name} {direction}"

    )





# ============================================================

# OBJECT COLOR

# ============================================================



def get_object_color(object_name):



    return OBJECT_COLORS.get(

        object_name,

        OBJECT_COLORS["default"]

    )





# ============================================================

# ALERT COOLDOWN

# ============================================================



def alert_allowed(

    track_id,

    alert_message

):



    global last_global_alert_time



    current_time = time.time()



    if (

        current_time -

        last_global_alert_time

        <

        GLOBAL_ALERT_COOLDOWN

    ):



        return False



    if track_id in last_alert_time:



        elapsed = (

            current_time -

            last_alert_time[track_id]

        )



        if elapsed < ALERT_COOLDOWN:



            return False



    return True






# ============================================================
# VALIDATION / EXPERIMENT LOGGING
# ============================================================
#
# This section was added without removing or replacing the
# original Lumora perception / CADE pipeline.
#
# It provides:
#   1. Runtime FPS measurement
#   2. Frame-processing latency measurement
#   3. Model inference timing
#   4. Alert-event logging
#   5. Manual functional-test marking
#   6. False-alert / missed-hazard counters
#   7. Session summary generation
#
# IMPORTANT:
#   - YOLO Precision / Recall / mAP are obtained from the
#     separate held-out validation dataset.
#   - The runtime metrics below evaluate the deployed
#     prototype, not the detector's ground-truth accuracy.
#   - Functional-test labels are manually marked by the
#     evaluator using H/N during controlled experiments.
# ============================================================

import csv
import os
from datetime import datetime
from statistics import mean, median


# ============================================================
# VALIDATION SETTINGS
# ============================================================

VALIDATION_OUTPUT_DIR = "results"

VALIDATION_SESSION_NAME = "lumora_validation"

PERFORMANCE_LOG_INTERVAL = 10

FPS_SMOOTHING_WINDOW = 30

TEST_ALERT_MEMORY_SECONDS = 10.0

# Prevent one held key from being counted as multiple tests.
TEST_KEY_COOLDOWN_SECONDS = 1.0

SHOW_VALIDATION_OVERLAY = True

WRITE_FRAME_PERFORMANCE_LOG = True

WRITE_EVENT_LOG = True

WRITE_TEST_LOG = True

WRITE_SESSION_SUMMARY = True


# ============================================================
# VALIDATION OUTPUT PATHS
# ============================================================

os.makedirs(
    VALIDATION_OUTPUT_DIR,
    exist_ok=True
)

validation_session_timestamp = (
    datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
)

validation_session_prefix = (
    f"{VALIDATION_SESSION_NAME}_"
    f"{validation_session_timestamp}"
)

performance_log_path = os.path.join(
    VALIDATION_OUTPUT_DIR,
    f"{validation_session_prefix}_performance.csv"
)

event_log_path = os.path.join(
    VALIDATION_OUTPUT_DIR,
    f"{validation_session_prefix}_events.csv"
)

test_log_path = os.path.join(
    VALIDATION_OUTPUT_DIR,
    f"{validation_session_prefix}_functional_tests.csv"
)

summary_log_path = os.path.join(
    VALIDATION_OUTPUT_DIR,
    f"{validation_session_prefix}_summary.txt"
)


# ============================================================
# VALIDATION RUNTIME STATE
# ============================================================

validation_frame_count = 0

validation_start_time = time.perf_counter()

validation_frame_times = deque(
    maxlen=FPS_SMOOTHING_WINDOW
)

validation_processing_times_ms = deque(
    maxlen=FPS_SMOOTHING_WINDOW
)

validation_general_times_ms = deque(
    maxlen=FPS_SMOOTHING_WINDOW
)

validation_road_times_ms = deque(
    maxlen=FPS_SMOOTHING_WINDOW
)

validation_total_alerts = 0

validation_last_alert_time = 0.0

validation_last_alert_text = ""

validation_last_alert_data = None

validation_alert_memory_until = 0.0

validation_last_test_key_time = 0.0


# ============================================================
# FUNCTIONAL TEST COUNTERS
# ============================================================

validation_test_total = 0

validation_true_positive = 0

validation_false_positive = 0

validation_true_negative = 0

validation_false_negative = 0


# ============================================================
# LAST FRAME STATE
# ============================================================

validation_last_highest_priority = None

validation_last_alert_triggered = False

validation_last_frame_fps = 0.0

validation_last_processing_ms = 0.0

validation_last_general_ms = 0.0

validation_last_road_ms = 0.0


# ============================================================
# CSV INITIALIZATION
# ============================================================

def initialize_validation_logs():
    """
    Create the validation CSV files and write their headers.

    A separate set of files is created for every execution
    session so that independent experiments are not mixed.
    """

    if WRITE_FRAME_PERFORMANCE_LOG:

        with open(
            performance_log_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as performance_file:

            writer = csv.writer(
                performance_file
            )

            writer.writerow(
                [
                    "timestamp",
                    "frame",
                    "fps",
                    "frame_processing_ms",
                    "general_yolo_ms",
                    "road_yolo_ms",
                    "detections",
                    "hazard_candidates",
                    "priority_object",
                    "priority_score"
                ]
            )


    if WRITE_EVENT_LOG:

        with open(
            event_log_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as event_file:

            writer = csv.writer(
                event_file
            )

            writer.writerow(
                [
                    "timestamp",
                    "frame",
                    "event",
                    "track_id",
                    "object",
                    "confidence",
                    "position",
                    "proximity",
                    "movement",
                    "cade_score",
                    "risk",
                    "alert"
                ]
            )


    if WRITE_TEST_LOG:

        with open(
            test_log_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as test_file:

            writer = csv.writer(
                test_file
            )

            writer.writerow(
                [
                    "timestamp",
                    "test_number",
                    "expected_condition",
                    "actual_alert",
                    "classification",
                    "object",
                    "position",
                    "proximity",
                    "movement",
                    "cade_score",
                    "alert_message"
                ]
            )


# ============================================================
# VALIDATION FILE HELPERS
# ============================================================

def validation_timestamp():
    """
    Return a timestamp suitable for experiment logs.
    """

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]


def validation_safe_text(value):
    """
    Convert a value to safe text for CSV output.
    """

    if value is None:

        return ""

    return str(value)


# ============================================================
# PERFORMANCE CALCULATION
# ============================================================

def calculate_validation_fps():
    """
    Calculate smoothed runtime FPS from recent frame times.
    """

    if not validation_frame_times:

        return 0.0

    average_frame_time = mean(
        validation_frame_times
    )

    if average_frame_time <= 0:

        return 0.0

    return (
        1.0
        / average_frame_time
    )


def calculate_average_processing_ms():
    """
    Calculate recent average frame-processing latency.
    """

    if not validation_processing_times_ms:

        return 0.0

    return mean(
        validation_processing_times_ms
    )


def calculate_average_general_ms():
    """
    Calculate recent average general YOLO inference time.
    """

    if not validation_general_times_ms:

        return 0.0

    return mean(
        validation_general_times_ms
    )


def calculate_average_road_ms():
    """
    Calculate recent average road-model inference time.
    """

    if not validation_road_times_ms:

        return 0.0

    return mean(
        validation_road_times_ms
    )


# ============================================================
# PERFORMANCE LOGGING
# ============================================================

def log_validation_performance(
    frame_number,
    fps,
    processing_ms,
    general_ms,
    road_ms,
    detection_count,
    hazard_count,
    priority_object,
    priority_score
):
    """
    Save periodic runtime measurements.

    Logging every frame is intentionally avoided because
    CPU inference can create unnecessarily large CSV files.
    """

    if not WRITE_FRAME_PERFORMANCE_LOG:

        return

    if (
        frame_number
        % PERFORMANCE_LOG_INTERVAL
        != 0
    ):

        return

    with open(
        performance_log_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as performance_file:

        writer = csv.writer(
            performance_file
        )

        writer.writerow(
            [
                validation_timestamp(),
                frame_number,
                f"{fps:.3f}",
                f"{processing_ms:.3f}",
                f"{general_ms:.3f}",
                f"{road_ms:.3f}",
                detection_count,
                hazard_count,
                validation_safe_text(
                    priority_object
                ),
                priority_score
            ]
        )


# ============================================================
# EVENT LOGGING
# ============================================================

def log_validation_event(
    event_name,
    detection=None,
    alert_message=""
):
    """
    Record an important CADE or alert event.
    """

    if not WRITE_EVENT_LOG:

        return

    if detection is None:

        row = [
            validation_timestamp(),
            validation_frame_count,
            event_name,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            alert_message
        ]

    else:

        row = [
            validation_timestamp(),
            validation_frame_count,
            event_name,
            detection.get(
                "track_id",
                ""
            ),
            detection.get(
                "object",
                ""
            ),
            f"{detection.get('confidence', 0.0):.3f}",
            detection.get(
                "position",
                ""
            ),
            detection.get(
                "proximity",
                ""
            ),
            detection.get(
                "movement",
                ""
            ),
            detection.get(
                "score",
                ""
            ),
            detection.get(
                "risk",
                ""
            ),
            alert_message
        ]

    with open(
        event_log_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as event_file:

        writer = csv.writer(
            event_file
        )

        writer.writerow(row)


# ============================================================
# ALERT STATE REGISTRATION
# ============================================================

def register_validation_alert(
    alert_message,
    alert_data
):
    """
    Register a generated alert for validation.

    A short memory window lets the evaluator press H after
    hearing the alert rather than on the exact video frame.
    """

    global validation_total_alerts

    global validation_last_alert_time

    global validation_last_alert_text

    global validation_last_alert_data

    global validation_alert_memory_until

    validation_total_alerts += 1

    validation_last_alert_time = (
        time.perf_counter()
    )

    validation_last_alert_text = (
        alert_message
    )

    validation_last_alert_data = (
        alert_data.copy()
    )

    validation_alert_memory_until = (
        time.perf_counter()
        + TEST_ALERT_MEMORY_SECONDS
    )


def validation_alert_is_recent():
    """
    Check whether a recent Lumora alert exists.
    """

    return (
        time.perf_counter()
        <= validation_alert_memory_until
    )


# ============================================================
# FUNCTIONAL TEST MARKING
# ============================================================

def record_functional_test(
    expected_hazard,
    actual_alert
):
    """
    Manually classify one controlled functional test.

    H = expected hazard
    N = expected no hazard

    The actual result is based on the recent Lumora alert
    state recorded by register_validation_alert().
    """

    global validation_test_total

    global validation_true_positive

    global validation_false_positive

    global validation_true_negative

    global validation_false_negative

    validation_test_total += 1

    if expected_hazard:

        if actual_alert:

            classification = (
                "TRUE_POSITIVE"
            )

            validation_true_positive += 1

        else:

            classification = (
                "FALSE_NEGATIVE"
            )

            validation_false_negative += 1

    else:

        if actual_alert:

            classification = (
                "FALSE_POSITIVE"
            )

            validation_false_positive += 1

        else:

            classification = (
                "TRUE_NEGATIVE"
            )

            validation_true_negative += 1


    detection = (
        validation_last_alert_data
        if actual_alert
        else None
    )


    if detection is None:

        object_name = ""

        position = ""

        proximity = ""

        movement = ""

        score = ""

        alert_message = ""

    else:

        object_name = detection.get(
            "object",
            ""
        )

        position = detection.get(
            "position",
            ""
        )

        proximity = detection.get(
            "proximity",
            ""
        )

        movement = detection.get(
            "movement",
            ""
        )

        score = detection.get(
            "score",
            ""
        )

        alert_message = (
            validation_last_alert_text
        )


    with open(
        test_log_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as test_file:

        writer = csv.writer(
            test_file
        )

        writer.writerow(
            [
                validation_timestamp(),
                validation_test_total,
                (
                    "HAZARD"
                    if expected_hazard
                    else "NO_HAZARD"
                ),
                (
                    "ALERT"
                    if actual_alert
                    else "NO_ALERT"
                ),
                classification,
                object_name,
                position,
                proximity,
                movement,
                score,
                alert_message
            ]
        )


    return classification


# ============================================================
# FUNCTIONAL TEST METRICS
# ============================================================

def calculate_functional_precision():
    """
    Calculate precision for manually marked end-to-end tests.
    """

    denominator = (
        validation_true_positive
        + validation_false_positive
    )

    if denominator == 0:

        return 0.0

    return (
        validation_true_positive
        / denominator
    )


def calculate_functional_recall():
    """
    Calculate recall for manually marked hazard tests.
    """

    denominator = (
        validation_true_positive
        + validation_false_negative
    )

    if denominator == 0:

        return 0.0

    return (
        validation_true_positive
        / denominator
    )


def calculate_functional_accuracy():
    """
    Calculate accuracy over manually labeled tests.
    """

    if validation_test_total == 0:

        return 0.0

    correct = (
        validation_true_positive
        + validation_true_negative
    )

    return (
        correct
        / validation_test_total
    )


def calculate_false_alert_rate():
    """
    Calculate false-alert rate among evaluated no-hazard
    scenarios.
    """

    denominator = (
        validation_true_negative
        + validation_false_positive
    )

    if denominator == 0:

        return 0.0

    return (
        validation_false_positive
        / denominator
    )


def calculate_missed_hazard_rate():
    """
    Calculate missed-hazard rate among evaluated hazard
    scenarios.
    """

    denominator = (
        validation_true_positive
        + validation_false_negative
    )

    if denominator == 0:

        return 0.0

    return (
        validation_false_negative
        / denominator
    )


# ============================================================
# SUMMARY GENERATION
# ============================================================

def write_validation_summary():
    """
    Write a human-readable validation summary at shutdown.
    """

    total_runtime = (
        time.perf_counter()
        - validation_start_time
    )

    average_fps = (
        calculate_validation_fps()
    )

    average_processing = (
        calculate_average_processing_ms()
    )

    average_general = (
        calculate_average_general_ms()
    )

    average_road = (
        calculate_average_road_ms()
    )

    functional_precision = (
        calculate_functional_precision()
    )

    functional_recall = (
        calculate_functional_recall()
    )

    functional_accuracy = (
        calculate_functional_accuracy()
    )

    false_alert_rate = (
        calculate_false_alert_rate()
    )

    missed_hazard_rate = (
        calculate_missed_hazard_rate()
    )


    with open(
        summary_log_path,
        "w",
        encoding="utf-8"
    ) as summary_file:

        summary_file.write(
            "LUMORA VALIDATION SUMMARY\n"
        )

        summary_file.write(
            "=" * 60
            + "\n\n"
        )

        summary_file.write(
            "Session information\n"
        )

        summary_file.write(
            "-" * 60
            + "\n"
        )

        summary_file.write(
            f"Session: "
            f"{validation_session_prefix}\n"
        )

        summary_file.write(
            f"Runtime seconds: "
            f"{total_runtime:.2f}\n"
        )

        summary_file.write(
            f"Frames processed: "
            f"{validation_frame_count}\n"
        )

        summary_file.write(
            f"Alerts generated: "
            f"{validation_total_alerts}\n\n"
        )

        summary_file.write(
            "Runtime performance\n"
        )

        summary_file.write(
            "-" * 60
            + "\n"
        )

        summary_file.write(
            f"Average smoothed FPS: "
            f"{average_fps:.3f}\n"
        )

        summary_file.write(
            f"Average frame processing: "
            f"{average_processing:.3f} ms\n"
        )

        summary_file.write(
            f"Average general YOLO: "
            f"{average_general:.3f} ms\n"
        )

        summary_file.write(
            f"Average road YOLO: "
            f"{average_road:.3f} ms\n\n"
        )

        summary_file.write(
            "Manual functional validation\n"
        )

        summary_file.write(
            "-" * 60
            + "\n"
        )

        summary_file.write(
            f"Total tests: "
            f"{validation_test_total}\n"
        )

        summary_file.write(
            f"True positives: "
            f"{validation_true_positive}\n"
        )

        summary_file.write(
            f"False positives: "
            f"{validation_false_positive}\n"
        )

        summary_file.write(
            f"True negatives: "
            f"{validation_true_negative}\n"
        )

        summary_file.write(
            f"False negatives: "
            f"{validation_false_negative}\n\n"
        )

        summary_file.write(
            f"Functional precision: "
            f"{functional_precision:.4f}\n"
        )

        summary_file.write(
            f"Functional recall: "
            f"{functional_recall:.4f}\n"
        )

        summary_file.write(
            f"Functional accuracy: "
            f"{functional_accuracy:.4f}\n"
        )

        summary_file.write(
            f"False-alert rate: "
            f"{false_alert_rate:.4f}\n"
        )

        summary_file.write(
            f"Missed-hazard rate: "
            f"{missed_hazard_rate:.4f}\n\n"
        )

        summary_file.write(
            "Interpretation note\n"
        )

        summary_file.write(
            "-" * 60
            + "\n"
        )

        summary_file.write(
            "Functional metrics are based only on manually "
            "marked controlled scenarios.\n"
        )

        summary_file.write(
            "They are not substitutes for detector validation "
            "metrics from the held-out dataset.\n"
        )

        summary_file.write(
            "YOLO Precision, Recall, mAP@50 and mAP@50-95 "
            "come from the detector validation set.\n"
        )


# ============================================================
# VALIDATION CONSOLE STATUS
# ============================================================

def print_validation_status():
    """
    Print a compact runtime validation status block.
    """

    print()

    print(
        "[VALIDATION STATUS]"
    )

    print(
        f"Frames: "
        f"{validation_frame_count}"
    )

    print(
        f"FPS: "
        f"{validation_last_frame_fps:.2f}"
    )

    print(
        f"Frame latency: "
        f"{validation_last_processing_ms:.2f} ms"
    )

    print(
        f"General YOLO: "
        f"{validation_last_general_ms:.2f} ms"
    )

    print(
        f"Road YOLO: "
        f"{validation_last_road_ms:.2f} ms"
    )

    print(
        f"Alerts: "
        f"{validation_total_alerts}"
    )

    print(
        f"Tests: "
        f"{validation_test_total}"
    )


# ============================================================
# VALIDATION OVERLAY
# ============================================================

def draw_validation_overlay(
    frame,
    fps,
    processing_ms,
    general_ms,
    road_ms
):
    """
    Draw runtime validation information on the camera view.
    """

    if not SHOW_VALIDATION_OVERLAY:

        return frame


    overlay_x = 20

    overlay_y = 85

    line_height = 20


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (
            overlay_x,
            overlay_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        f"LATENCY: {processing_ms:.0f}ms",
        (
            overlay_x,
            overlay_y + line_height
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        f"GEN: {general_ms:.0f}ms",
        (
            overlay_x,
            overlay_y + line_height * 2
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        f"ROAD: {road_ms:.0f}ms",
        (
            overlay_x,
            overlay_y + line_height * 3
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        f"ALERTS: {validation_total_alerts}",
        (
            overlay_x,
            overlay_y + line_height * 4
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        (
            f"TESTS: {validation_test_total} "
            f"TP:{validation_true_positive} "
            f"FP:{validation_false_positive}"
        ),
        (
            overlay_x,
            overlay_y + line_height * 5
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        (
            "H=hazard test | "
            "N=no-hazard test | "
            "Q=quit"
        ),
        (
            overlay_x,
            overlay_y + line_height * 6
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1
    )


    return frame


# ============================================================
# VALIDATION HELP
# ============================================================

def print_validation_instructions():
    """
    Explain the manual test keys before the experiment.
    """

    print()

    print(
        "=" * 60
    )

    print(
        "LUMORA VALIDATION CONTROLS"
    )

    print(
        "=" * 60
    )

    print(
        "H -> Mark current scenario as EXPECTED HAZARD"
    )

    print(
        "N -> Mark current scenario as EXPECTED NO HAZARD"
    )

    print(
        "Q -> Stop the prototype"
    )

    print()

    print(
        "Use H after placing a genuine hazard in the scene."
    )

    print(
        "Use N when testing a non-hazard / safe situation."
    )

    print()

    print(
        "The functional metrics are based on these manual"
    )

    print(
        "labels and should only be reported for controlled"
    )

    print(
        "scenarios that were actually tested."
    )

    print(
        "=" * 60
    )


# ============================================================
# VALIDATION CLEANUP
# ============================================================

def finalize_validation():
    """
    Save the final validation summary and print the evidence
    file locations.
    """

    if WRITE_SESSION_SUMMARY:

        write_validation_summary()


    print()

    print(
        "=" * 60
    )

    print(
        "LUMORA VALIDATION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Performance log: "
        f"{performance_log_path}"
    )

    print(
        f"Event log: "
        f"{event_log_path}"
    )

    print(
        f"Functional tests: "
        f"{test_log_path}"
    )

    print(
        f"Summary: "
        f"{summary_log_path}"
    )

    print()

    print(
        f"Total tests: "
        f"{validation_test_total}"
    )

    print(
        f"TP: "
        f"{validation_true_positive}"
    )

    print(
        f"FP: "
        f"{validation_false_positive}"
    )

    print(
        f"TN: "
        f"{validation_true_negative}"
    )

    print(
        f"FN: "
        f"{validation_false_negative}"
    )

    print(
        f"Functional precision: "
        f"{calculate_functional_precision():.3f}"
    )

    print(
        f"Functional recall: "
        f"{calculate_functional_recall():.3f}"
    )

    print(
        f"Functional accuracy: "
        f"{calculate_functional_accuracy():.3f}"
    )

    print(
        f"False-alert rate: "
        f"{calculate_false_alert_rate():.3f}"
    )

    print(
        f"Missed-hazard rate: "
        f"{calculate_missed_hazard_rate():.3f}"
    )

    print(
        "=" * 60
    )


# ============================================================
# END VALIDATION / EXPERIMENT LOGGING
# ============================================================


# ============================================================

# MAIN LOOP

# ============================================================



print()

print("Starting Lumora...")

print("Press Q to quit.")

print()





frame_counter = 0


# ============================================================
# START VALIDATION SESSION
# ============================================================

initialize_validation_logs()

print_validation_instructions()

print(
    f"Validation output directory: "
    f"{VALIDATION_OUTPUT_DIR}"
)

print(
    f"Session: {validation_session_prefix}"
)

print()


while True:

    frame_loop_start = time.perf_counter()

    validation_last_alert_triggered = False

    validation_frame_count += 1

    success, frame = cap.read()



    if not success:



        print(

            "ERROR: Could not read camera frame."

        )



        break



    frame_counter += 1



    frame_height, frame_width = (

        frame.shape[:2]

    )



    frame_area = (

        frame_width *

        frame_height

    )





    # ========================================================

    # GENERAL YOLO + BYTETRACK

    # ========================================================



    general_inference_start = time.perf_counter()

    general_results = general_model.track(



        frame,



        persist=True,



        tracker="bytetrack.yaml",



        conf=GENERAL_CONFIDENCE,



        imgsz=IMG_SIZE,



        device="cpu",



        verbose=False

    )



    general_result = general_results[0]

    general_inference_ms = (
        time.perf_counter()
        - general_inference_start
    ) * 1000.0

    validation_general_times_ms.append(
        general_inference_ms
    )






    # ========================================================

    # DETECTION STORAGE

    # ========================================================



    detections = []





    # ========================================================

    # GENERAL MODEL DETECTIONS

    # ========================================================



    if (

        general_result.boxes is not None

        and general_result.boxes.is_track

    ):



        boxes = (

            general_result.boxes.xyxy

            .cpu()

            .numpy()

        )



        track_ids = (

            general_result.boxes.id

            .int()

            .cpu()

            .tolist()

        )



        class_ids = (

            general_result.boxes.cls

            .int()

            .cpu()

            .tolist()

        )



        confidences = (

            general_result.boxes.conf

            .cpu()

            .numpy()

        )



        for (

            box,

            track_id,

            class_id,

            confidence

        ) in zip(

            boxes,

            track_ids,

            class_ids,

            confidences

        ):



            object_name = (

                general_model.names[class_id]

            )



            x1, y1, x2, y2 = map(

                int,

                box

            )



            detections.append({



                "box": (

                    x1,

                    y1,

                    x2,

                    y2

                ),



                "object": object_name,



                "confidence": float(

                    confidence

                ),



                "track_id": int(

                    track_id

                ),



                "source": "general"



            })





    # ========================================================

    # ROAD MODEL

    # ========================================================



    road_inference_ms = 0.0

    if frame_counter % ROAD_MODEL_INTERVAL == 0:

        road_inference_start = time.perf_counter()



        road_results = road_model.predict(



            frame,



            conf=ROAD_CONFIDENCE,



            imgsz=IMG_SIZE,



            device="cpu",



            verbose=False

        )



        road_result = road_results[0]

        road_inference_ms = (
            time.perf_counter()
            - road_inference_start
        ) * 1000.0

        validation_road_times_ms.append(
            road_inference_ms
        )






        if road_result.boxes is not None:



            road_boxes = (

                road_result.boxes.xyxy

                .cpu()

                .numpy()

            )



            road_class_ids = (

                road_result.boxes.cls

                .int()

                .cpu()

                .tolist()

            )



            road_confidences = (

                road_result.boxes.conf

                .cpu()

                .numpy()

            )





            for (

                road_box,

                road_class_id,

                road_confidence

            ) in zip(

                road_boxes,

                road_class_ids,

                road_confidences

            ):



                original_class = (

                    road_model.names[

                        road_class_id

                    ]

                )



                # ------------------------------------------------

                # Ignore fire hydrants

                # ------------------------------------------------



                if (

                    original_class

                    not in ROAD_CLASS_MAPPING

                ):



                    continue





                object_name = ROAD_CLASS_MAPPING[

                    original_class

                ]



                road_box = tuple(

                    map(

                        int,

                        road_box

                    )

                )





                # ------------------------------------------------

                # Duplicate check

                # ------------------------------------------------



                duplicate = False



                for existing in detections:



                    if (

                        existing["object"]

                        != object_name

                    ):



                        continue



                    iou = calculate_iou(



                        road_box,



                        existing["box"]

                    )



                    if (

                        iou

                        >

                        IOU_DUPLICATE_THRESHOLD

                    ):



                        duplicate = True



                        break





                if duplicate:



                    continue





                # ------------------------------------------------

                # Create road track

                # ------------------------------------------------



                road_track_id = (

                    assign_road_track_id(

                        object_name,

                        road_box

                    )

                )





                detections.append({



                    "box": road_box,



                    "object": object_name,



                    "confidence": float(

                        road_confidence

                    ),



                    "track_id": road_track_id,



                    "source": "road"



                })





    cleanup_road_tracks()





    # ========================================================

    # PROCESS ALL FUSED DETECTIONS

    # ========================================================



    current_hazards = []





    for detection in detections:



        x1, y1, x2, y2 = (

            detection["box"]

        )



        object_name = detection["object"]

        track_id = detection["track_id"]

        confidence = detection["confidence"]



        box_area = (

            (x2 - x1) *

            (y2 - y1)

        )



        x_center = (

            x1 + x2

        ) / 2



        y_center = (

            y1 + y2

        ) / 2





        # ====================================================

        # POSITION

        # ====================================================



        position = get_position(

            x_center,

            frame_width

        )





        # ====================================================

        # VISUAL PROXIMITY

        # ====================================================



        proximity = get_visual_proximity(

            box_area,

            frame_area

        )





        # ====================================================

        # TRACK HISTORY

        # ====================================================



        track_area_history[

            track_id

        ].append(

            box_area

        )



        track_position_history[

            track_id

        ].append(

            (x_center, y_center)

        )



        track_seen_time[

            track_id

        ] = time.time()





        # ====================================================

        # MOVEMENT

        # ====================================================



        movement = calculate_movement(

            track_id

        )





        # ====================================================

        # CADE

        # ====================================================



        score, risk = calculate_cade(



            position,



            proximity,



            movement,



            object_name

        )





        # ====================================================

        # ALERT CANDIDATE

        # ====================================================



        candidate = is_alert_candidate(



            object_name,



            position,



            proximity,



            movement,



            score

        )





        current_hazards.append({



            "track_id": track_id,



            "object": object_name,



            "confidence": confidence,



            "position": position,



            "proximity": proximity,



            "movement": movement,



            "score": score,



            "risk": risk,



            "candidate": candidate



        })





        # ====================================================

        # DRAW

        # ====================================================



        object_color = get_object_color(

            object_name

        )



        cv2.rectangle(



            frame,



            (x1, y1),



            (x2, y2),



            object_color,



            2

        )





        label = (



            f"{object_name} "



            f"ID:{track_id} "



            f"{confidence:.2f}"

        )





        cv2.putText(



            frame,



            label,



            (

                x1,

                max(y1 - 10, 20)

            ),



            cv2.FONT_HERSHEY_SIMPLEX,



            0.55,



            object_color,



            2

        )





        cv2.putText(



            frame,



            f"{position} | {proximity}",



            (

                x1,

                y2 + 20

            ),



            cv2.FONT_HERSHEY_SIMPLEX,



            0.50,



            object_color,



            2

        )





        cv2.putText(



            frame,



            f"MOVEMENT: {movement}",



            (

                x1,

                y2 + 42

            ),



            cv2.FONT_HERSHEY_SIMPLEX,



            0.50,



            (255, 255, 255),



            2

        )





        cv2.putText(



            frame,



            f"RISK: {risk} | SCORE: {score}",



            (

                x1,

                y2 + 64

            ),



            cv2.FONT_HERSHEY_SIMPLEX,



            0.50,



            (0, 165, 255),



            2

        )





    # ========================================================

    # FIND ALERT CANDIDATES

    # ========================================================



    alert_candidates = [



        hazard



        for hazard in current_hazards



        if hazard["candidate"]



    ]





    # ========================================================

    # PRIORITIZE

    # ========================================================



    highest_priority = None





    if alert_candidates:



        alert_candidates.sort(



            key=lambda x: (



                x["score"],



                x["proximity"] == "NEAR",



                x["movement"] == "APPROACHING"



            ),



            reverse=True

        )



        highest_priority = (

            alert_candidates[0]

        )





    # ========================================================

    # ALERT + TTS

    # ========================================================



    if highest_priority is not None:



        track_id = (

            highest_priority[

                "track_id"

            ]

        )



        alert = generate_alert(



            highest_priority["object"],



            highest_priority["position"],



            highest_priority["movement"]

        )





        if alert_allowed(



            track_id,



            alert



        ):



            current_time = time.time()



            last_alert_time[

                track_id

            ] = current_time



            last_global_alert_time = (

                current_time

            )



            last_alert_message[

                track_id

            ] = alert





            print()



            print(

                "[CADE ALERT]"

            )



            print(

                f"ID: {track_id}"

            )



            print(

                f"Object: "

                f"{highest_priority['object']}"

            )



            print(

                f"Position: "

                f"{highest_priority['position']}"

            )



            print(

                f"Proximity: "

                f"{highest_priority['proximity']}"

            )



            print(

                f"Movement: "

                f"{highest_priority['movement']}"

            )



            print(

                f"Risk: "

                f"{highest_priority['risk']}"

            )



            print(

                f"Score: "

                f"{highest_priority['score']}"

            )



            print(

                f"Alert: {alert}"

            )





            validation_last_alert_triggered = True

            register_validation_alert(
                alert,
                highest_priority
            )

            log_validation_event(
                "CADE_ALERT",
                highest_priority,
                alert
            )

            speak_alert(alert)





            cv2.putText(



                frame,



                "VOICE ALERT",



                (

                    20,

                    frame_height - 75

                ),



                cv2.FONT_HERSHEY_SIMPLEX,



                0.65,



                (0, 165, 255),



                2

            )





            cv2.putText(



                frame,



                alert.upper(),



                (

                    20,

                    frame_height - 40

                ),



                cv2.FONT_HERSHEY_SIMPLEX,



                0.70,



                (0, 165, 255),



                2

            )



    else:



        cv2.putText(



            frame,



            "NO IMMEDIATE HAZARD",



            (

                20,

                frame_height - 40

            ),



            cv2.FONT_HERSHEY_SIMPLEX,



            0.65,



            (255, 255, 255),



            2

        )





    # ========================================================

    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    frame_processing_ms = (
        time.perf_counter()
        - frame_loop_start
    ) * 1000.0

    frame_duration_seconds = (
        frame_processing_ms
        / 1000.0
    )

    validation_frame_times.append(
        frame_duration_seconds
    )

    validation_processing_times_ms.append(
        frame_processing_ms
    )

    validation_last_processing_ms = (
        frame_processing_ms
    )

    validation_last_general_ms = (
        general_inference_ms
    )

    validation_last_road_ms = (
        road_inference_ms
    )

    validation_last_frame_fps = (
        calculate_validation_fps()
    )

    if highest_priority is None:

        priority_object_name = ""

        priority_score_value = ""

    else:

        priority_object_name = (
            highest_priority["object"]
        )

        priority_score_value = (
            highest_priority["score"]
        )


    log_validation_performance(
        validation_frame_count,
        validation_last_frame_fps,
        frame_processing_ms,
        general_inference_ms,
        road_inference_ms,
        len(detections),
        len(alert_candidates),
        priority_object_name,
        priority_score_value
    )


    if (
        validation_frame_count
        % 100
        == 0
    ):

        print_validation_status()


    # HEADER

    # ========================================================



    cv2.putText(



        frame,



        "LUMORA - EDGE AI VISION",



        (20, 30),



        cv2.FONT_HERSHEY_SIMPLEX,



        0.75,



        (255, 255, 255),



        2

    )





    cv2.putText(



        frame,



        "YOLO11s + Road YOLO + ByteTrack + CADE",



        (20, 58),



        cv2.FONT_HERSHEY_SIMPLEX,



        0.50,



        (255, 255, 255),



        2

    )





    # ========================================================

    # ========================================================
    # VALIDATION OVERLAY
    # ========================================================

    draw_validation_overlay(
        frame,
        validation_last_frame_fps,
        validation_last_processing_ms,
        validation_last_general_ms,
        validation_last_road_ms
    )


    # DISPLAY

    # ========================================================



    cv2.imshow(



        "Lumora - Edge AI Prototype",



        frame

    )





    # ========================================================

    # EXIT

    # ========================================================



    key = cv2.waitKey(1) & 0xFF



    # ========================================================
    # MANUAL FUNCTIONAL TEST CONTROLS
    # ========================================================

    if key == ord("h"):

        current_test_key_time = time.perf_counter()
        if current_test_key_time - validation_last_test_key_time < TEST_KEY_COOLDOWN_SECONDS:
            continue
        validation_last_test_key_time = current_test_key_time

        actual_alert = (
            validation_alert_is_recent()
        )

        classification = record_functional_test(
            True,
            actual_alert
        )

        print()

        print(
            "[FUNCTIONAL TEST]"
        )

        print(
            "Expected: HAZARD"
        )

        print(
            f"Actual: "
            f"{'ALERT' if actual_alert else 'NO ALERT'}"
        )

        print(
            f"Result: {classification}"
        )


    elif key == ord("n"):

        current_test_key_time = time.perf_counter()
        if current_test_key_time - validation_last_test_key_time < TEST_KEY_COOLDOWN_SECONDS:
            continue
        validation_last_test_key_time = current_test_key_time

        actual_alert = (
            validation_alert_is_recent()
        )

        classification = record_functional_test(
            False,
            actual_alert
        )

        print()

        print(
            "[FUNCTIONAL TEST]"
        )

        print(
            "Expected: NO HAZARD"
        )

        print(
            f"Actual: "
            f"{'ALERT' if actual_alert else 'NO ALERT'}"
        )

        print(
            f"Result: {classification}"
        )


    if key == ord("q"):



        break





# ============================================================
# FINALIZE VALIDATION
# ============================================================

finalize_validation()


# ============================================================

# CLEANUP

# ============================================================



cap.release()



cv2.destroyAllWindows()



print()

print("Lumora stopped.")