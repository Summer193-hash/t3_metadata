import os
import numpy as np
import json
from tqdm import tqdm
import os.path as osp
import shutil

# --- Configuration ---
# The root folder of the processed NPZ dataset
NPZ_ROOT = 'C:\\Users\\Sumer Singh\\Desktop\\metadata\\tekken_dataset_npz'
# The final metadata file
OUTPUT_MASTER_METADATA_FILE = osp.join(NPZ_ROOT, 'metadata.json')
# The subfolders to be scanned
FOLDERS_TO_SCAN = ['P1_WIN', 'P2_WIN', 'DRAW']
# The in-game round time in seconds (for duration calculation)
GAME_MAX_ROUND_SECONDS = 40 
# the FPS the game was recorded at
RECORDED_FPS = 30

# --- Static Information ---
STATIC_GAME_INFO = {
    "stage": "Taekwondo Dojo"
}
# HARDCODING IT FOR NOW. REMINDER: UPDATE THE MAIN DATA COLLECTION PIPELINE TO INCLUDE THE CHARACTER NAMES
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

def reorganize_and_generate_metadata():
    """
    Scans for loose NPZ files OR existing round folders, reorganizes
    as needed, and generates/updates all metadata.
    This script is safe to run multiple times.
    """
    
    round_entries_for_master_file = []
    max_length = 0
    
    print(f"Scanning and reorganizing {NPZ_ROOT}...")

    for folder_name in FOLDERS_TO_SCAN:
        folder_path = osp.join(NPZ_ROOT, folder_name)
        
        if not osp.isdir(folder_path):
            print(f"Warning: Folder not found, skipping: {folder_path}")
            continue

        print(f"Scanning folder: {folder_name}...")
        
        # Determine the outcome from the folder name
        if folder_name == 'P1_WIN':
            winner_str = 'p1'
        elif folder_name == 'P2_WIN':
            winner_str = 'p2'
        else:
            winner_str = 'draw'

        # We must use list() to snapshot the directory contents
        # before we start moving files around.
        try:
            items_in_folder = list(os.listdir(folder_path))
        except FileNotFoundError:
            print(f"Warning: Directory {folder_path} does not exist.")
            continue

        for item_name in tqdm(items_in_folder, desc=f"Processing {folder_name}", unit="item"):
            item_path = osp.join(folder_path, item_name)
            
            npz_path_to_process = None
            round_id_str = None
            filename = None
            current_round_folder_path = None # This is where the per-round JSON will go

            # Case 1: Item is a loose .npz file (e.g., 'round_003.npz')
            # Needs to be reorganized.
            if osp.isfile(item_path) and item_name.endswith('.npz') and item_name.startswith('round_'):
                filename = item_name
                round_id_str = filename.replace('.npz', '')
                old_npz_path = item_path
                
                print(f"\nReorganizing: {filename}")
                
                # 1. Creating a New Folder Structure
                current_round_folder_path = osp.join(folder_path, round_id_str)
                os.makedirs(current_round_folder_path, exist_ok=True)
                
                # 2. Moving the NPZ File inside it
                new_npz_path = osp.join(current_round_folder_path, filename)
                try:
                    shutil.move(old_npz_path, new_npz_path)
                except Exception as e:
                    print(f"  Error moving file {old_npz_path} to {new_npz_path}: {e}")
                    continue
                
                npz_path_to_process = new_npz_path 
                
            # Case 2: Item is an existing round directory (e.g., 'round_001/')
            # Already organized, just needs to be processed.
            elif osp.isdir(item_path) and item_name.startswith('round_'):
                round_id_str = item_name
                filename = f"{round_id_str}.npz"
                current_round_folder_path = item_path # The folder already exists
                
                npz_path_to_process = osp.join(current_round_folder_path, filename)
                
                if not osp.isfile(npz_path_to_process):
                    # This check is important
                    print(f"\nWarning: Found folder {item_path} but no {filename} inside. Skipping.")
                    continue
            
            # Case 3: Item is something else (e.g., 'metadata.json', '.DS_Store', etc.)
            else:
                # We just ignore this item and continue the loop
                continue

            # 3. Reading Data from New Location
            # This block (and the rest of the loop) is now common logic for both cases
            try:
                with np.load(npz_path_to_process) as data:
                    valid_frames_array = data['valid_frames']
                    states_array = data['states'] # (max_len, 3)
                    
                    # Get length and padding info
                    original_length = int(np.sum(valid_frames_array))
                    current_padded_length = len(valid_frames_array)
                    
                    if current_padded_length > max_length:
                        max_length = current_padded_length
                    
                    # Get game state info
                    initial_health_p1 = 1.0
                    initial_health_p2 = 1.0
                    initial_timer_norm = 1.0
                    
                    final_state = states_array[original_length - 1]
                    final_health_p1 = float(final_state[0])
                    final_health_p2 = float(final_state[1])
                    final_timer_norm = float(final_state[2])
                    
                    # Get file info
                    file_size_bytes = osp.getsize(npz_path_to_process)
                    
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

                # 4. Assembling Per-Round Metadata
                # Note: 'npz_path' is now just the filename, as it's relative to the metadata.json file in the same folder
                per_round_metadata_entry = {
                    "round_id": round_id_str,
                    "npz_path": filename,
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
                        "file_size_bytes": file_size_bytes
                    },
                    "game_state": {
                        "initial_health": {"p1": initial_health_p1, "p2": initial_health_p2},
                        "final_health": {"p1": final_health_p1, "p2": final_health_p2},
                        "timer": {"initial": initial_timer_norm, "final": final_timer_norm}
                    }
                }
                
                # 5. Save Per-Round metadata.json File
                per_round_metadata_path = osp.join(current_round_folder_path, 'metadata.json')
                with open(per_round_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(per_round_metadata_entry, f, indent=4)

                # 6. Prepare Entry for Master File
                # This entry needs the FULL relative path from the root
                master_entry = per_round_metadata_entry.copy()
                master_entry["npz_path"] = osp.relpath(npz_path_to_process, NPZ_ROOT).replace(os.sep, '/')
                
                round_entries_for_master_file.append(master_entry)
                
            except Exception as e:
                print(f"\nError reading or processing {npz_path_to_process}: {e}")

    if not round_entries_for_master_file:
        print("\nNo .npz files found or processed. No metadata file created.")
        return

    print(f"\nScan and reorganization complete. Found {len(round_entries_for_master_file)} rounds.")
    print(f"Determined max sequence length: {max_length}")

    # --- 7. Sorting the round entries ---
    print("Sorting round entries by round_id...")
    round_entries_for_master_file.sort(key=lambda x: x['round_id'])
    print("Sorting complete.")

    # 8. Assembling the Master Metadata File
    master_metadata = {
        "dataset_name": "Tekken 3 Processed NPZ Dataset",
        "description": "Padded rounds of gameplay. Each NPZ contains padded arrays for images, actions, and states.",
        "max_sequence_length": max_length,
        "frame_dimensions": STATIC_FRAME_INFO,
        "npz_array_keys": {
            "images": f"Padded (T, C, H, W) video frames. Shape: ({max_length}, {STATIC_FRAME_INFO['channels']}, {STATIC_FRAME_INFO['height']}, {STATIC_FRAME_INFO['width']})",
            "actions_p1": f"Padded (T,) integer action IDs for Player 1. Shape: ({max_length},)",
            "actions_p2": f"Padded (T,) integer action IDs for Player 2. Shape: ({max_length},)",
            "states": f"Padded (T, 3) normalized game states [P1_Health, P2_Health, Timer]. Shape: ({max_length}, 3)",
            "valid_frames": f"Padded (T,) boolean mask. 1 = real frame, 0 = padding. Shape: ({max_length},)"
        },
        "action_id_p1_mapping": {
            "description": "Integer ID from 0-255 calculated as a sum of powers of 2",
            "columns": BUTTON_COLUMNS_P1,
            "bit_values": {col: int(POWERS_OF_2[i]) for i, col in enumerate(BUTTON_COLUMNS_P1)}
        },
        "action_id_p2_mapping": {
            "description": "Integer ID from 0-255 calculated as a sum of powers of 2",
            "columns": BUTTON_COLUMNS_P2,
            "bit_values": {col: int(POWERS_OF_2[i]) for i, col in enumerate(BUTTON_COLUMNS_P2)}
        },
        "rounds": round_entries_for_master_file
    }
    
    # 9. Saving the final MASTER JSON file
    try:
        with open(OUTPUT_MASTER_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(master_metadata, f, indent=4)
        print(f"\nSuccess! Master metadata saved to: {OUTPUT_MASTER_METADATA_FILE}")
    except Exception as e:
        print(f"\nError saving master metadata: {e}")

if __name__ == "__main__":
    reorganize_and_generate_metadata()
