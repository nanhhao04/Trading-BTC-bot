import os
import shutil

# Define roots
ROOT_DIR = r"C:\Users\Admin\PycharmProjects\BotTrading-RF"
SCRATCH_DIR = os.path.join(ROOT_DIR, "scratch")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

os.makedirs(SCRATCH_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def safe_move(src_path, dest_path):
    if os.path.exists(src_path):
        try:
            # Overwrite if exists
            if os.path.exists(dest_path):
                if os.path.isdir(dest_path):
                    shutil.rmtree(dest_path)
                else:
                    os.remove(dest_path)
            shutil.move(src_path, dest_path)
            print(f"Moved: {os.path.basename(src_path)} -> {os.path.relpath(dest_path, ROOT_DIR)}")
        except Exception as e:
            print(f"Error moving {os.path.basename(src_path)}: {e}")
    else:
        print(f"Not found: {os.path.basename(src_path)} (Skipping)")

print("--- START REPO REORGANIZATION ---")

# 1. Move scratch/diagnostic scripts to scratch/
safe_move(os.path.join(ROOT_DIR, "analyze_reward_stats.py"), os.path.join(SCRATCH_DIR, "analyze_reward_stats.py"))
safe_move(os.path.join(ROOT_DIR, "reward_analysis_results.txt"), os.path.join(SCRATCH_DIR, "reward_analysis_results.txt"))
safe_move(os.path.join(ROOT_DIR, "live_bot.py"), os.path.join(SCRATCH_DIR, "live_bot_demo.py"))

# 2. Move temp balance csv to logs/
safe_move(os.path.join(ROOT_DIR, "balance_history.csv"), os.path.join(LOGS_DIR, "balance_history.csv"))
safe_move(os.path.join(ROOT_DIR, "src", "balance_history.csv"), os.path.join(LOGS_DIR, "balance_history_src_temp.csv"))

# 3. Move markdown analysis files to docs/
safe_move(os.path.join(ROOT_DIR, "TRAINING_ANALYSIS.md"), os.path.join(DOCS_DIR, "TRAINING_ANALYSIS.md"))
safe_move(os.path.join(ROOT_DIR, "FIXES_SUMMARY.md"), os.path.join(DOCS_DIR, "FIXES_SUMMARY.md"))

print("--- FINISHED REPO REORGANIZATION ---")
