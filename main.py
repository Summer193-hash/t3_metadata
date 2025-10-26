import os
import numpy as np
import json
from tqdm import tqdm
import os.path as osp
import shutil
import cv2

# --- Configuration ---
# The root folder of the ORIGINAL processed NPZ dataset (SOURCE)
NPZ_ROOT = './tekken_dataset_npz'
# The root folder for the NEW video dataset (DESTINATION)
OUTPUT_ROOT = './data_mp4'
# The final metadata file (will be saved in the OUTPUT_ROOT)
OUTPUT_MASTER_METADATA_FILE = osp.join(OUTPUT_ROOT, 'metadata.json')
# The subfolders to be scanned in the SOURCE directory
FOLDERS_TO_SCAN = ['P1_WIN', 'P2_WIN']
# The in-game round time in seconds (for duration calculation)
GAME_MAX_ROUND_SECONDS = 40
# the FPS the game was recorded at (for video and calculations)
RECORDED_FPS = 30

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


def convert_npz_to_video_dataset():
    """
    Scans the original NPZ_ROOT, converts each round into a video (mp4)
    and saves it to a new clean directory structure in OUTPUT_ROOT.
    
    Generates per-round and master metadata.json files.
    This script is safe to run multiple times (it will overwrite existing files).
    """
    
    round_entries_for_master_file = []
    
    print(f"Scanning source {NPZ_ROOT} and writing to {OUTPUT_ROOT}")
    
    # Ensure the main output directory exists
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for folder_name in FOLDERS_TO_SCAN:
        folder_path = osp.join(NPZ_ROOT, folder_name)
        
        if not osp.isdir(folder_path):
            print(f"Warning: Source folder not found, skipping: {folder_path}")
            continue

        print(f"Scanning source folder: {folder_name}")
        
        # Determine the outcome from the folder name
        if folder_name == 'P1_WIN':
            winner_str = 'p1'
        elif folder_name == 'P2_WIN':
            winner_str = 'p2'
        else:
            winner_str = 'draw'

        try:
            items_in_folder = list(os.listdir(folder_path))
        except FileNotFoundError:
            print(f"Warning: Directory {folder_path} does not exist.")
            continue

        for item_name in tqdm(items_in_folder, desc=f"Converting {folder_name}", unit="item"):
            item_path = osp.join(folder_path, item_name)
            
            npz_path_to_process = None
            round_id_str = None

            # Case 1: Item is a loose .npz file (e.g., 'round_003.npz')
            if osp.isfile(item_path) and item_name.endswith('.npz') and item_name.startswith('round_'):
                round_id_str = item_name.replace('.npz', '')
                npz_path_to_process = item_path
                
            # Case 2: Item is an existing round directory (e.g., 'round_001/')
            elif osp.isdir(item_path) and item_name.startswith('round_'):
                round_id_str = item_name
                filename = f"{round_id_str}.npz"
                npz_path_to_process = osp.join(item_path, filename)
                
                if not osp.isfile(npz_path_to_process):
                    print(f"\nWarning: Found folder {item_path} but no {filename} inside. Skipping.")
                    continue
            
            # Case 3: Item is something else (e.g., 'metadata.json', etc.)
            else:
                continue

            # Define NEW Output Paths
            current_output_round_folder = osp.join(OUTPUT_ROOT, round_id_str)
            os.makedirs(current_output_round_folder, exist_ok=True)
            
            output_video_path = osp.join(current_output_round_folder, 'video.mp4')
            output_metadata_path = osp.join(current_output_round_folder, 'metadata.json')

            try:
                # Reading Data from SOURCE NPZ
                with np.load(npz_path_to_process) as data:
                    valid_frames_mask = data['valid_frames']
                    original_length = int(np.sum(valid_frames_mask))
                    
                    # Slice only the data we need
                    valid_images = data['images'][:original_length] # (T, C, H, W)
                    valid_states = data['states'][:original_length] # (T, 3)
                    # We read states just for the metadata, but don't save it

                # Write Video File
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, RECORDED_FPS, 
                                                   (STATIC_FRAME_INFO["width"], STATIC_FRAME_INFO["height"]))
                    
                    for frame_c_h_w in valid_images:
                        # Convert (C, H, W) -> (H, W, C) for OpenCV
                        frame_h_w_c = frame_c_h_w.transpose(1, 2, 0)
                        # Convert RGB -> BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame_h_w_c, cv2.COLOR_RGB2BGR)
                        video_writer.write(frame_bgr.astype(np.uint8))
                        
                    video_writer.release()
                except Exception as e:
                    print(f"\nError writing video for {round_id_str}: {e}")
                    continue # Skip this round

                # Get Game State & Duration Info (from loaded data)
                initial_health_p1 = 1.0
                initial_health_p2 = 1.0
                initial_timer_norm = 1.0
                
                final_state = valid_states[-1] # Get last valid state
                final_health_p1 = float(final_state[0])
                final_health_p2 = float(final_state[1])
                final_timer_norm = float(final_state[2])
                
                # Get file info
                video_file_size_bytes = osp.getsize(output_video_path)
                
                # Calculate durations
                game_duration_seconds = (initial_timer_norm - final_timer_norm) * GAME_MAX_ROUND_SECONDS
                clip_duration_seconds = original_length / RECORDED_FPS
                
                # Determine win method
                if final_timer_norm <= 0.01:
                    method = "timeout"
                else:
                    method = "knockout"
                    
                # Parse round number
                try:
                    round_num_int = int(round_id_str.split('_')[-1])
                except:
                    round_num_int = -1

                # Assembling Per-Round Metadata
                per_round_metadata_entry = {
                    "round_id": round_id_str,
                    "video_path": "video.mp4", # Relative to this JSON
                    "players": {
                        "p1": { 
                            "character": STATIC_PLAYER_INFO["p1_character"],
                            "position": "left" 
                        },
                        "p2": { 
                            "character": STATIC_PLAYER_INFO["p2_character"],
                            "position": "right" 
                        }
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
                        "video_file_size_bytes": video_file_size_bytes
                    },
                    "game_state": {
                        "initial_health": {"p1": initial_health_p1, "p2": initial_health_p2},
                        "final_health": {"p1": final_health_p1, "p2": final_health_p2},
                        "timer": {"initial": initial_timer_norm, "final": final_timer_norm}
                    }
                }
                
                # Save Per-Round metadata.json File
                with open(output_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(per_round_metadata_entry, f, indent=4)

                # Prepare Entry for Master File
                master_entry = per_round_metadata_entry.copy()
                # Update paths to be relative to the MASTER file
                master_entry["video_path"] = f"{round_id_str}/video.mp4"
                master_entry["local_metadata_path"] = f"{round_id_str}/metadata.json"
                
                round_entries_for_master_file.append(master_entry)
                
            except Exception as e:
                print(f"\nError reading or processing {npz_path_to_process}: {e}")

    if not round_entries_for_master_file:
        print("\nNo .npz files found or processed. No metadata file created.")
        return

    print(f"\nScan and conversion complete. Processed {len(round_entries_for_master_file)} rounds.")

    # Sorting the round entries 
    print("Sorting round entries by round_id/")
    round_entries_for_master_file.sort(key=lambda x: x['round_id'])
    print("Sorting complete.")

    # Assembling the Master Metadata File
    master_metadata = {
        "dataset_name": "Tekken 3 Video Dataset",
        "description": "Each round folder (e.g., 'round_001/') contains an 'video.mp4' and a 'metadata.json' file.",
        "frame_dimensions": STATIC_FRAME_INFO,
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
    
    # Saving the final MASTER JSON file
    try:
        with open(OUTPUT_MASTER_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(master_metadata, f, indent=4)
        print(f"\nSuccess! Master metadata saved to: {OUTPUT_MASTER_METADATA_FILE}")
    except Exception as e:
        print(f"\nError saving master metadata: {e}")

if __name__ == "__main__":
    convert_npz_to_video_dataset()