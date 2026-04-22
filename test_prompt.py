import subprocess
import sys
import threading
import argparse
from transformers import AutoTokenizer
import os
import time

def read_stderr(proc, ready_event):
    for line in iter(proc.stderr.readline, b''):
        line = line.decode('utf-8', errors='replace').strip()
        if not line: continue
        print(f"\033[90m{line}\033[0m", file=sys.stderr)
        if "[Inference] Engine ready." in line:
            ready_event.set()

def main():
    engine_path = "./engine.exe" if sys.platform == "win32" else "./engine"
    if not os.path.exists(engine_path):
        engine_path = "build_output/engine.exe"
    
    tokenizer_path = "./tokenizer"
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "Qwen/Qwen2.5-32B"
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    print("Starting engine...")
    proc = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )
    
    ready_event = threading.Event()
    stderr_thread = threading.Thread(target=read_stderr, args=(proc, ready_event))
    stderr_thread.daemon = True
    stderr_thread.start()
    
    print("Waiting for engine to load model (this may take up to 1200 seconds)...")
    if not ready_event.wait(timeout=1200):
        print("Error: Engine timed out during startup.")
        proc.terminate()
        return

    prompt = "Hi, can you introduce yourself?"
    print(f"\n\033[92mYou: {prompt}\033[0m")
    
    # Fix: Missing Instruct Chat Template
    messages = [{"role": "user", "content": prompt}]
    chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(chat_input)
    token_str = " ".join(map(str, tokens)) + " \n"
    print(f"Sending tokens: {token_str}")
    proc.stdin.write(token_str.encode('utf-8'))
    proc.stdin.flush()
    
    print("\033[96mQwen 3.6: \033[0m", end='', flush=True)
    
    # Read response
    current_tokens = []
    buffer = ""
    start_time = time.time()
    while True:
        if proc.poll() is not None: break
        
        char = proc.stdout.read(1)
        if not char: break
        char = char.decode('utf-8', errors='ignore')
        
        if char == ' ' or char == '\n':
            if buffer:
                try:
                    token_val = int(buffer)
                    current_tokens.append(token_val)
                    full_text = tokenizer.decode(current_tokens)
                    prev_text = tokenizer.decode(current_tokens[:-1]) if len(current_tokens) > 1 else ""
                    new_text = full_text[len(prev_text):]
                    print(new_text, end='', flush=True)
                except ValueError:
                    pass
                buffer = ""
            if char == '\n':
                print("\n", flush=True)
                break
        else:
            buffer += char
        
        if time.time() - start_time > 60:
            print("\n[Test Timeout]")
            break

    proc.terminate()
    proc.wait()
    print("\nTest finished.")

if __name__ == "__main__":
    main()
