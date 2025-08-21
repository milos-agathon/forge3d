import hashlib
from forge3d import Renderer

def render_bytes():
    r = Renderer(256,256)
    return r.render_triangle_rgba().tobytes()

b1 = render_bytes()
b2 = render_bytes()

h1 = hashlib.sha256(b1).hexdigest()
h2 = hashlib.sha256(b2).hexdigest()
print("sha256:", h1, h2)
assert h1 == h2, "non-deterministic bytes across two renders"
print("OK: same bytes across two renders")
