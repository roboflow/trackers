import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class InteractivePMapViewer:
    def __init__(self, frames, p_maps, grid_size=50, alpha=0.4, show_text=True, cmap='Greens'):
        self.frames = frames
        self.p_maps = p_maps
        self.grid_size = grid_size
        self.alpha = alpha
        self.show_text = show_text
        self.cmap = cmap
        self.idx = 0

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.im = None
        self.texts = []
        self.patches = []

        self.render()

        plt.show()

    def render(self):
        self.ax.clear()
        frame = self.frames[self.idx]
        p_map = self.p_maps[self.idx]
        h, w = frame.shape[:2]

        self.ax.imshow(frame)
        normed = np.clip(p_map, 0.0, 1.0)
        grid_h, grid_w = p_map.shape

        self.patches.clear()
        self.texts.clear()

        for gy in range(grid_h):
            for gx in range(grid_w):
                prob = normed[gy, gx]
                if prob > 0:
                    x = gx * self.grid_size
                    y = gy * self.grid_size
                    rect = patches.Rectangle((x, y), self.grid_size, self.grid_size,
                                             linewidth=0.5, edgecolor='white',
                                             facecolor=plt.cm.get_cmap(self.cmap)(prob),
                                             alpha=self.alpha)
                    self.ax.add_patch(rect)
                    self.patches.append(rect)

                    if self.show_text:
                        text = self.ax.text(x + self.grid_size / 2,
                                            y + self.grid_size / 2,
                                            f"{prob:.2f}",
                                            ha='center', va='center',
                                            fontsize=6, color='black')
                        self.texts.append(text)

        # Grid lines
        for gx in range(0, w, self.grid_size):
            self.ax.axvline(gx, color='white', lw=0.3, alpha=0.5)
        for gy in range(0, h, self.grid_size):
            self.ax.axhline(gy, color='white', lw=0.3, alpha=0.5)

        self.ax.set_title(f"Frame {self.idx + 1}/{len(self.frames)}")
        self.ax.set_xlim([0, w])
        self.ax.set_ylim([h, 0])
        self.ax.axis('off')
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.idx = min(self.idx + 1, len(self.frames) - 1)
            self.render()
        elif event.key == 'left':
            self.idx = max(self.idx - 1, 0)
            self.render()
