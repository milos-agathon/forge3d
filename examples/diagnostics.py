import vulkan_forge as vf

def main():
    print("Adapters:", vf.enumerate_adapters())
    print("Probe:", vf.device_probe())

if __name__ == "__main__":
    main()