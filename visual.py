import os
import re
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pipeline import PersonSearchPipeline


def parse_cam(cam_name):
    """
    Parse strings like 'c3s2' → (2,1) (0‑indexed grid coordinates).
    """
    m = re.match(r'c(\d+)s(\d+)', cam_name)
    if not m:
        raise ValueError(f"Unexpected camera format: {cam_name}")
    col = int(m.group(1)) - 1
    row = int(m.group(2)) - 1
    return (col, row)


def animate(path):
    coords = [parse_cam(p['camera']) for p in path]
    xs, ys = zip(*coords)

    fig, ax = plt.subplots()
    ax.set_xlim(min(xs)-1, max(xs)+1)
    ax.set_ylim(min(ys)-1, max(ys)+1)
    point, = ax.plot([], [], 'o', markersize=12)

    def update(i):
        x, y = coords[i]
        point.set_data(x, y)
        return point,

    ani = FuncAnimation(fig, update, frames=len(coords),
                        interval=1000, blit=True, repeat=False)
    ax.set_xlabel("Camera column (c# - 1)")
    ax.set_ylabel("Scene row (s# - 1)")
    ax.set_title("Person Movement Path")
    plt.show()


def main():
    print("=== Multi‑Camera Path Visualization ===")
    desc = input("Description: ")

    # Default camera folders, only include directories
    cam_root = "data/Market1501/cams"
    if not os.path.isdir(cam_root):
        print(f"Error: '{cam_root}' not found")
        return
    cams = [d for d in sorted(os.listdir(cam_root))
            if not d.startswith('.') and os.path.isdir(os.path.join(cam_root, d))]
    cam_paths = [os.path.join(cam_root, c) for c in cams]

    print("\nRunning search & track…")
    ps = PersonSearchPipeline()
    res = ps.search_and_track(desc, camera_dirs=cam_paths)
    ps.save_snapshots(res)

    # Animate only if we got a valid path
    if 'path' in res and res['path']:
        animate(res['path'])
    else:
        print("No path to visualize. Check your description or thresholds.")


if __name__ == "__main__":
    main()
