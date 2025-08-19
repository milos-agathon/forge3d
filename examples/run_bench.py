from vulkan_forge import run_benchmark

def main():
    res = run_benchmark("renderer_rgba", width=256, height=256, iterations=20)
    print("stats:", res["stats"])
    print("throughput:", res["throughput"])

if __name__ == "__main__":
    main()