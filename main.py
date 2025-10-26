import os
import numpy as np
import json
from tqdm import tqdm
import os.path as osp
import shutil
import cv2
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---
# The root folder of the ORIGINAL processed NPZ dataset (SOURCE)
NPZ_ROOT = './tekken_dataset_npz'
# The root folder for the NEW video dataset (DESTINATION)
OUTPUT_ROOT = './data_mp4'
# The final metadata file (will be saved in the OUTPUT_ROOT)
OUTPUT_MASTER_METADATA_FILE = osp.join(OUTPUT_ROOT, 'metadata.json')
# The subfolders to be scanned in the SOURCE directory
FOLDERS_TO_SCAN = ['P1_WIN', 'P2_WIN', 'DRAW']
# The in-game round time in seconds (for duration calculation)
GAME_MAX_ROUND_SECONDS = 40
# the FPS the game was recorded at (for video and calculations)
RECORDED_FPS = 30
# --- Number of parallel CPU workers ---
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 2)
# --- NEW: Define the video codec string ---
VIDEO_CODEC_STR = 'avc1' # H.264 codec


# --- Static Information ---
STATIC_GAME_INFO = {
    "stage": "Taekwondo Dojo"
}
STATIC_PLAYER_INFO = {
    "p1_character": "Hwoarang",
    "p2_character": "Jin"
}
STATIC_FRAME_INFO = {
    "width": 736,
    "height": 448,
    "channels": 3,
    "fps": RECORDED_FPS
}

BUTTON_COLUMNS_P1 = ['P1_Up', 'P1_Down', 'P1_Left', 'P1_Right', 'P1_Square', 'P1_Triangle', 'P1_Cross', 'P1_Circle']
BUTTON_COLUMNS_P2 = ['P2_Up', 'P2_Down', 'P2_Left', 'P2_Right', 'P2_Square', 'P2_Triangle', 'P2_Cross', 'P2_Circle']
POWERS_OF_2 = 2**np.arange(7, -1, -1, dtype=np.uint16)


def process_round(task):
    """
    Worker function to process a single round.
    Reads an NPZ, writes a video, and generates a local metadata.json.
    Returns the entry for the master metadata file on success, or None on failure.
    """
    npz_path_to_process, round_id_str, winner_str = task

    # --- 1. Define NEW Output Paths ---
    current_output_round_folder = osp.join(OUTPUT_ROOT, round_id_str)
    os.makedirs(current_output_round_folder, exist_ok=True)
    
    output_video_path = osp.join(current_output_round_folder, 'video.mp4')
    output_metadata_path = osp.join(current_output_round_folder, 'metadata.json')

    try:
        # 2. Reading Data from SOURCE NPZ
        with np.load(npz_path_to_process) as data:
            valid_frames_mask = data['valid_frames']
            original_length = int(np.sum(valid_frames_mask))
            
            if original_length == 0:
                print(f"\nWarning: {round_id_str} has 0 valid frames. Skipping video creation.")
                return None 
            
            valid_images = data['images'][:original_length] # (T, C, H, W)
            valid_states = data['states'][:original_length] # (T, 3)

        # 3. --- Write Video File ---
        try:
            # --- Use the codec constant ---
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC_STR) 
            video_writer = cv2.VideoWriter(output_video_path, fourcc, RECORDED_FPS, 
                                           (STATIC_FRAME_INFO["width"], STATIC_FRAME_INFO["height"]))
            
            if not video_writer.isOpened():
                print(f"\nError: VideoWriter failed to open for {round_id_str}. Check codec/permissions.")
                return None

            for frame_c_h_w in valid_images:
                frame_h_w_c = frame_c_h_w.transpose(1, 2, 0)
                frame_bgr = cv2.cvtColor(frame_h_w_c, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr.astype(np.uint8))
                
            video_writer.release()
        
        except Exception as e:
            print(f"\nError writing video for {round_id_str}: {e}")
            return None 

        # 4. Get Game State & Duration Info
        initial_health_p1 = 1.0
        initial_health_p2 = 1.0
        initial_timer_norm = 1.0
        
        final_state = valid_states[-1]
        final_health_p1 = float(final_state[0])
        final_health_p2 = float(final_state[1])
        final_timer_norm = float(final_state[2])
        
        video_file_size_bytes = osp.getsize(output_video_path)
        game_duration_seconds = (initial_timer_norm - final_timer_norm) * GAME_MAX_ROUND_SECONDS
        clip_duration_seconds = original_length / RECORDED_FPS
        
        method = "timeout" if final_timer_norm <= 0.01 else "knockout"
            
        try:
            round_num_int = int(round_id_str.split('_')[-1])
        except:
            round_num_int = -1

        # 5. Assembling Per-Round Metadata
        per_round_metadata_entry = {
            "round_id": round_id_str,
            "video_path": "video.mp4",
            "players": {
                "p1": {"character": STATIC_PLAYER_INFO["p1_character"], "position": "left"},
                "p2": {"character": STATIC_PLAYER_INFO["p2_character"], "position": "right"}
            },
            "round_info": {
                "round_number": round_num_int,
                "stage": STATIC_GAME_INFO["stage"],
                "winner": winner_str,
                "method": method,
                "game_duration_seconds": round(game_duration_seconds, 2),
                "clip_duration_seconds": round(clip_duration_seconds, 2),
                "frame_count": original_length
            },
            "technical": {
                "fps": RECORDED_FPS,
                "resolution": [STATIC_FRAME_INFO["width"], STATIC_FRAME_INFO["height"]],
                # --- ADDED CODEC TO LOCAL METADATA ---
                "codec": VIDEO_CODEC_STR, 
                "video_file_size_bytes": video_file_size_bytes
            },
            "game_state": {
                "initial_health": {"p1": initial_health_p1, "p2": initial_health_p2},
                "final_health": {"p1": final_health_p1, "p2": final_health_p2},
                "timer": {"initial": initial_timer_norm, "final": final_timer_norm}
            }
        }
        
        # 6. Save Per-Round metadata.json File
        with open(output_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(per_round_metadata_entry, f, indent=4)

        # 7. Prepare Entry for Master File and return it
        master_entry = per_round_metadata_entry.copy()
        master_entry["video_path"] = f"{round_id_str}/video.mp4"
        master_entry["local_metadata_path"] = f"{round_id_str}/metadata.json"
        
        return master_entry
        
    except Exception as e:
        print(f"\nError reading or processing {npz_path_to_process}: {e}")
        return None


def convert_npz_to_video_dataset():
    """
    Scans the source NPZ_ROOT and uses a ProcessPoolExecutor to convert
    all rounds in parallel.
    """
    
    print(f"Scanning source {NPZ_ROOT}...")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    tasks_to_process = [] 

    # --- STAGE 1: Scan and Collect Tasks ---
    for folder_name in FOLDERS_TO_SCAN:
        folder_path = osp.join(NPZ_ROOT, folder_name)
        
        if not osp.isdir(folder_path):
            print(f"Warning: Source folder not found, skipping: {folder_path}")
            continue

        print(f"Scanning source folder: {folder_name}...")
        
        winner_str = 'draw'
        if folder_name == 'P1_WIN':
            winner_str = 'p1'
        elif folder_name == 'P2_WIN':
            winner_str = 'p2'

        try:
            items_in_folder = list(os.listdir(folder_path))
        except FileNotFoundError:
            print(f"Warning: Directory {folder_path} does not exist.")
            continue

        for item_name in items_in_folder:
            item_path = osp.join(folder_path, item_name)
            npz_path_to_process = None
            round_id_str = None

            if osp.isfile(item_path) and item_name.endswith('.npz') and item_name.startswith('round_'):
                round_id_str = item_name.replace('.npz', '')
                npz_path_to_process = item_path
            elif osp.isdir(item_path) and item_name.startswith('round_'):
                round_id_str = item_name
                filename = f"{round_id_str}.npz"
                npz_path_to_process = osp.join(item_path, filename)
                if not osp.isfile(npz_path_to_process):
                    continue
            else:
                continue
            
            tasks_to_process.append((npz_path_to_process, round_id_str, winner_str))

    if not tasks_to_process:
        print("\nNo .npz files found to process. No metadata file created.")
        return

    # --- STAGE 2: Execute Tasks in Parallel ---
    print(f"\nFound {len(tasks_to_process)} rounds to process.")
    print(f"Starting parallel conversion using up to {MAX_WORKERS} workers...")
    
    round_entries_for_master_file = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_round, task): task for task in tasks_to_process}
        
        for future in tqdm(as_completed(futures), total=len(tasks_to_process), desc="Converting rounds"):
            result = future.result()
            if result:
                round_entries_for_master_file.append(result)
            else:
                task = futures[future]
                print(f"Warning: Task failed and returned no data for {task[1]}")

    if not round_entries_for_master_file:
        print("\nAll tasks failed. No metadata file created.")
        return
        
    print(f"\nConversion complete. Successfully processed {len(round_entries_for_master_file)} rounds.")

    # --- STAGE 3: Sorting and Final Metadata Generation ---
    print("Sorting round entries by round_id...")
    round_entries_for_master_file.sort(key=lambda x: x['round_id'])
    print("Sorting complete.")

    # 10. Assembling the Master Metadata File
    master_metadata = {
        "dataset_name": "Tekken 3 Video Dataset",
        "description": "Each round folder (e.g., 'round_001/') contains an 'video.mp4' and a 'metadata.json' file.",
        "frame_dimensions": {
            "width": STATIC_FRAME_INFO["width"],
            "height": STATIC_FRAME_INFO["height"],
            "channels": STATIC_FRAME_INFO["channels"],
            "fps": STATIC_FRAME_INFO["fps"],
            # --- ADDED CODEC TO MASTER METADATA ---
            "codec": VIDEO_CODEC_STR 
        },
        "action_id_p1_mapping": {
            "description": "Integer ID from 0-255 calculated as a sum of powers of 2 (for context, not in files)",
            "columns": BUTTON_COLUMNS_P1,
            "bit_values": {col: int(POWERS_OF_2[i]) for i, col in enumerate(BUTTON_COLUMNS_P1)}
        },
        "action_id_p2_mapping": {
            "description": "Integer ID from 0-255 calculated as a sum of powers of 2 (for context, not in files)",
            "columns": BUTTON_COLUMNS_P2,
            "bit_values": {col: int(POWERS_OF_2[i]) for i, col in enumerate(BUTTON_COLUMNS_P2)}
        },
        "rounds": round_entries_for_master_file
    }
    
    # 11. Saving the final MASTER JSON file
    try:
        with open(OUTPUT_MASTER_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(master_metadata, f, indent=4)
        print(f"\nSuccess! Master metadata saved to: {OUTPUT_MASTER_METADATA_FILE}")
    except Exception as e:
        print(f"\nError saving master metadata: {e}")

# This __name__ check is CRITICAL for multiprocessing to work correctly!
if __name__ == "__main__":
    convert_npz_to_video_dataset()