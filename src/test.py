import subprocess

command = [
    "python", "-m", "graphrag", "query",
    "--root", ".", "--method", "global",
    "--query", "how are elon and mark related?"
]

result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Output
print("STDOUT:")
print(result.stdout)

print("\nSTDERR:")
print(result.stderr)

# You can use these variables however you want
output_text = result.stdout
error_text = result.stderr