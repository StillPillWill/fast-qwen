import subprocess
import sys
import threading
import argparse
from transformers import AutoTokenizer
import os

def read_stderr(proc, show_metrics):
    for line in iter(proc.stderr.readline, b''):
        line = line.decode('utf-8', errors='replace').strip()
        if not line: continue
        # Always show FATAL errors
        if "[FATAL]" in line:
            print(f"\033[91m{line}\033[0m", file=sys.stderr)
        elif show_metrics:
            # Show other metrics
            print(f"\033[90m{line}\033[0m", file=sys.stderr)
        elif "[Inference]" in line or "[Model]" in line or "[Main]" in line:
            # Show progress even without --show-metrics
            print(f"\033[90m{line}\033[0m", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Chat CLI for MoE Engine")
    parser.add_argument("--show-metrics", action="store_true", help="Print telemetry overlaid", default=True)
    args = parser.parse_args()

    engine_path = "./engine.exe" if sys.platform == "win32" else "./engine"
    if not os.path.exists(engine_path):
        engine_path = "build_output/engine.exe"
    
    tokenizer_path = "./tokenizer"
    if not os.path.exists(tokenizer_path):
        print(f"Error: Local tokenizer directory '{tokenizer_path}' not found. Falling back to Qwen/Qwen2.5-32B.")
        tokenizer_path = "Qwen/Qwen2.5-32B"
    else:
        print(f"Loading local Qwen 3.6 Tokenizer from {tokenizer_path}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    print("Starting engine...", flush=True)
    
    try:
        proc = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
    except FileNotFoundError:
        print(f"Error: Could not find '{engine_path}'. Please build the engine first.")
        sys.exit(1)
    
    stderr_thread = threading.Thread(target=read_stderr, args=(proc, args.show_metrics))
    stderr_thread.daemon = True
    stderr_thread.start()
    
    print("Type your message below. Type 'exit' or 'quit' to quit.\n")
    
    while True:
        try:
            user_input = input("\033[92mYou: \033[0m")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
            
            tokens = tokenizer.encode(user_input)
            if not tokens:
                continue
                
            # Send to engine
            token_str = " ".join(map(str, tokens)) + " \n"
            try:
                proc.stdin.write(token_str.encode('utf-8'))
                proc.stdin.flush()
            except OSError as e:
                print(f"\033[91mEngine Communication Error: {e}\033[0m")
                break
            
            print("\033[96mQwen 3.6: \033[0m", end='', flush=True)
            
            # Read response from engine
            buffer = ""
            while True:
                char = proc.stdout.read(1)
                if not char:
                    break
                char = char.decode('utf-8', errors='ignore')
                if char == ' ':
                    if buffer:
                        try:
                            token_str = tokenizer.decode([int(buffer)])
                            print(token_str, end='', flush=True)
                        except ValueError:
                            print(f" [T:{buffer}]", end='', flush=True)
                        buffer = ""
                    print(" ", end='', flush=True)
                elif char == '\n':
                    if buffer:
                        try:
                            token_str = tokenizer.decode([int(buffer)])
                            print(token_str, end='', flush=True)
                        except ValueError:
                            print(f" [T:{buffer}]", end='', flush=True)
                        buffer = ""
                    print("", flush=True)
                    break # Done with this turn
                else:
                    buffer += char
                    
        except (KeyboardInterrupt, EOFError):
            break

    proc.terminate()
    proc.wait()
    print("\nEngine exited.")

if __name__ == "__main__":
    main()
