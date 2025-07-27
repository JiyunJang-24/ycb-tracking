import os
import subprocess

def fastSAM_inference(
				text_prompt, img_path, output_path, script_path="FastSAM/Inference.py"):
		os.makedirs(output_path, exist_ok=True)

		# Runs the script and waits for it to finish
		result = subprocess.run(['python', script_path, '--text_prompt', text_prompt, '--img_path', img_path, '--output', output_path, '--model_path', 'FastSAM/weights/FastSAM.pt'], capture_output=True, text=True)
		if result.returncode != 0:
				print("Error running FastSAM inference:")
				print(result.stderr)
		else:
				print("FastSAM inference completed successfully.")
				print(result.stdout)
