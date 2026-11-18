import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Icon({ size = 16, children, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const CubeIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z" />
    <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" />
  </Icon>
);

export const CameraIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M4 8.5A2.5 2.5 0 016.5 6h5A2.5 2.5 0 0114 8.5v7A2.5 2.5 0 0111.5 18h-5A2.5 2.5 0 014 15.5v-7z" />
    <path d="M14 11l6-3.5v9L14 13z" />
  </Icon>
);

export const SunIcon = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </Icon>
);

export const EyeIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z" />
    <circle cx="12" cy="12" r="3" />
  </Icon>
);

export const EyeOffIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M3 3l18 18" />
    <path d="M10.6 6.1A9.9 9.9 0 0112 6c6 0 9.5 6 9.5 6a17.6 17.6 0 01-3.5 4" />
    <path d="M6.3 7.8A17.4 17.4 0 002.5 12S6 18 12 18a9.6 9.6 0 003.5-.7" />
    <path d="M9.9 9.9a3 3 0 004.2 4.2" />
  </Icon>
);

export const LockIcon = (props: IconProps) => (
  <Icon {...props}>
    <rect x="5" y="10.5" width="14" height="10" rx="2" />
    <path d="M8.5 10.5V7.5a3.5 3.5 0 017 0v3" />
  </Icon>
);

export const UnlockIcon = (props: IconProps) => (
  <Icon {...props}>
    <rect x="5" y="10.5" width="14" height="10" rx="2" />
    <path d="M8.5 10.5V7.5a3.5 3.5 0 016.8-1.2" />
  </Icon>
);

export const UndoIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M9 7H5V3" />
    <path d="M5 7l3.5-3.5" />
    <path d="M5.5 7H14a5.5 5.5 0 110 11H8" />
  </Icon>
);

export const RedoIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M15 7h4V3" />
    <path d="M19 7l-3.5-3.5" />
    <path d="M18.5 7H10a5.5 5.5 0 100 11h6" />
  </Icon>
);

export const SaveIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M5 3h11l3 3v15H5z" />
    <path d="M8 3v6h8V3M8 21v-7h8v7" />
  </Icon>
);

export const FolderIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M3 7a2 2 0 012-2h4l2 2.5h8a2 2 0 012 2V18a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
  </Icon>
);

export const FileIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M6 3h8l4 4v14H6z" />
    <path d="M14 3v4h4" />
  </Icon>
);

export const PlusIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M12 5v14M5 12h14" />
  </Icon>
);

export const SlidersIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M4 7h10M18 7h2M4 17h4M12 17h8" />
    <circle cx="16" cy="7" r="2" />
    <circle cx="10" cy="17" r="2" />
  </Icon>
);

export const ChevronIcon = ({ open, ...props }: IconProps & { open: boolean }) => (
  <Icon {...props}>
    <path d={open ? "M6 9l6 6 6-6" : "M9 6l6 6-6 6"} />
  </Icon>
);

export const CloseIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M6 6l12 12M18 6L6 18" />
  </Icon>
);

export const RestoreIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M4 12a8 8 0 108-8" />
    <path d="M4 4v5h5" />
  </Icon>
);

export const MountainIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M3 19h18L14.5 7l-3.2 5.2L9 9.5 3 19z" />
    <path d="M12.6 9.6l1.9-2.6 1.6 3" />
  </Icon>
);

export const MapIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M9 4.5L3 7v12.5l6-2.5 6 2.5 6-2.5V4.5L15 7 9 4.5z" />
    <path d="M9 4.5v12.5M15 7v12.5" />
  </Icon>
);

export const CloudIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M7 18h9a4 4 0 0 0 .6-7.96A5.5 5.5 0 0 0 6 11.5 3.25 3.25 0 0 0 7 18z" />
  </Icon>
);

export const KeyframeIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M3 12h5M16 12h5" />
    <path d="M12 6l4 6-4 6-4-6z" />
  </Icon>
);

export const CursorIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M5 3l6.5 16 2.2-6.1L20 10.5 5 3z" />
  </Icon>
);

export const FlyIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M22 2L11 13" />
    <path d="M22 2l-7 20-4-9-9-4z" />
  </Icon>
);

export const MoveIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M12 3v18M3 12h18" />
    <path d="M12 3l-2.4 2.6M12 3l2.4 2.6M12 21l-2.4-2.6M12 21l2.4-2.6M3 12l2.6-2.4M3 12l2.6 2.4M21 12l-2.6-2.4M21 12l-2.6 2.4" />
  </Icon>
);

export const RotateIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M20 12a8 8 0 11-2.6-5.9" />
    <path d="M20 4v4.5h-4.5" />
  </Icon>
);

export const ScaleIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M4 20L14 10" />
    <path d="M9 20H4v-5M20 9V4h-5" />
    <rect x="3" y="17" width="4" height="4" rx="1" />
    <rect x="17" y="3" width="4" height="4" rx="1" />
  </Icon>
);

export const TrashIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M4 7h16M9 7V4.5h6V7M6.5 7l.8 13h9.4l.8-13" />
    <path d="M10 11v5.5M14 11v5.5" />
  </Icon>
);

export const CopyIcon = (props: IconProps) => (
  <Icon {...props}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M15 9V6.5A2.5 2.5 0 0012.5 4h-5A2.5 2.5 0 005 6.5v5A2.5 2.5 0 007.5 14H9" />
  </Icon>
);

export const ImportIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M12 4v10" />
    <path d="M8 10.5l4 4 4-4" />
    <path d="M4 18.5h16" />
  </Icon>
);

export const objectIcon = (type: "mesh" | "terrain" | "camera" | "light") => {
  switch (type) {
    case "mesh":
      return CubeIcon;
    case "terrain":
      return MountainIcon;
    case "camera":
      return CameraIcon;
    case "light":
      return SunIcon;
  }
};
