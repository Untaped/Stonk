import os
import ast

def find_str_shadowing(root_dir):
    print(f"🔎 Scanning Python files under: {root_dir}\n")
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(subdir, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        source = f.read()
                        tree = ast.parse(source, filename=path)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Assign):
                                for target in node.targets:
                                    if isinstance(target, ast.Name) and target.id == "str":
                                        print(f"❌ Shadowing 'str' in assignment → {path}:{target.lineno}")
                            elif isinstance(node, ast.FunctionDef):
                                for arg in node.args.args:
                                    if arg.arg == "str":
                                        print(f"❌ Shadowing 'str' in function arg → {path}:{arg.lineno}")
                            elif isinstance(node, ast.Name) and node.id == "str":
                                # Detect if str is used in a suspicious way (like function call)
                                if isinstance(node.ctx, ast.Load):
                                    parent = getattr(node, 'parent', None)
                                    if isinstance(parent, ast.Call) and parent.func == node:
                                        print(f"❗ Possible call to shadowed 'str' → {path}:{node.lineno}")
                        # Add parent linking to catch suspicious calls
                        for node in ast.walk(tree):
                            for child in ast.iter_child_nodes(node):
                                child.parent = node
                except Exception as e:
                    print(f"⚠️ Error processing {path}: {e}")

if __name__ == "__main__":
    find_str_shadowing(".")  # current directory
