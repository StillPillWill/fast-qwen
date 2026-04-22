import subprocess
import sys
import threading
import argparse
from transformers import AutoTokenizer
import os
import time

def read_stderr(proc, show_metrics):
    try:
        for line in iter(proc.stderr.readline, b''):
            line = line.decode('utf-8', errors='replace').strip()
            if not line: continue
            if "[FATAL]" in line:
                print(f"\n\033[91m{line}\033[0m", file=sys.stderr)
            elif show_metrics:
                print(f"\033[90m{line}\033[0m", file=sys.stderr)
            elif any(x in line for x in ["[Inference]", "[Model]", "[Main]"]):
                print(f"\033[90m{line}\033[0m", file=sys.stderr)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Chat CLI for MoE Engine")
    parser.add_argument("--show-metrics", action="store_true", help="Print telemetry overlaid", default=False)
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
        if proc.poll() is not None:
            print("\033[91mEngine has exited unexpectedly.\033[0m")
            break
            
        try:
            user_input = input("\033[92mYou: \033[0m")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
            
            # Fix: Missing Instruct Chat Template
            messages = [{"role": "user", "content": user_input}]
            chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer.encode(chat_input)
            if not tokens:
                continue
                
            token_str = " ".join(map(str, tokens)) + " \n"
            try:
                proc.stdin.write(token_str.encode('utf-8'))
                proc.stdin.flush()
            except OSError:
                break
            
            print("\033[96mQwen 3.6: \033[0m", end='', flush=True)
            
            # Streaming decode
            current_tokens = []
            buffer = ""
            while True:
                # Fix: Subprocess Pipe Deadlock
                if proc.poll() is not None: break
                
                char = proc.stdout.read(1)
                if not char: break
                char = char.decode('utf-8', errors='ignore')
                
                if char in [' ', '\n']:
                    if buffer:
                        try:
                            token_id = int(buffer)
                            current_tokens.append(token_id)
                            # Correctly handle BPE by decoding the full sequence so far
                            # and only printing the diff
                            full_text = tokenizer.decode(current_tokens)
                            prev_text = tokenizer.decode(current_tokens[:-1]) if len(current_tokens) > 1 else ""
                            new_text = full_text[len(prev_text):]
                            print(new_text, end='', flush=True)
                        except ValueError:
                            pass
                        buffer = ""
                    if char == '\n':
                        print("", flush=True)
                        break
                else:
                    buffer += char
                    
        except (KeyboardInterrupt, EOFError):
            break

    if proc.poll() is None:
        proc.terminate()
    print("\nDone.")

if __name__ == "__main__":
    main()
