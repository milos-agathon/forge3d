import forge3d as f3d

def main():
    print("Adapters:", f3d.enumerate_adapters())
    print("Probe:", f3d.device_probe())

if __name__ == "__main__":
    main()