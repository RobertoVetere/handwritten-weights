"""Download stories15M (Karpathy's llama2.c checkpoint) and its tokenizer into src/, and verify SHA-256."""
import hashlib, os, sys, urllib.request

FILES = {
    'stories15M.bin': ('https://raw.githubusercontent.com/cgbur/llama2.zig/master/stories15M.bin',
                       'cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a'),
    'tokenizer.bin': ('https://raw.githubusercontent.com/tairov/llama2.mojo/master/tokenizer.bin',
                      '74f899535f8f99cdeac697474b697f3fa40467b8376dab7b6c4d8c753f28ae9f'),
}
dest = os.path.join(os.path.dirname(__file__), '..', 'src')
for name, (url, sha) in FILES.items():
    path = os.path.join(dest, name)
    if not os.path.exists(path):
        print(f'downloading {name} ...'); urllib.request.urlretrieve(url, path)
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    ok = h == sha
    print(f'{name}: sha256 {"OK" if ok else "MISMATCH " + h}')
    if not ok: sys.exit(1)
