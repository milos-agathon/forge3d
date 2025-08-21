import forge3d as f3d

def main():
    xy, uv, idx = f3d.grid_generate(32, 32, spacing=(1.0, 1.0), origin="center")
    print("grid:", xy.shape, uv.shape, idx.shape)

if __name__ == "__main__":
    main()