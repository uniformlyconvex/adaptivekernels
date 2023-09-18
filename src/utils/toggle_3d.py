import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.backend_bases import PickEvent
from matplotlib.collections import PathCollection
from matplotlib.text import Text

from typing import Iterable, Tuple, Optional, List, Dict
from dataclasses import dataclass

ArrayIsh = np.ndarray | torch.Tensor

@dataclass
class TogglableData:
    X: ArrayIsh
    Y: ArrayIsh
    alpha: float = 1.0
    name: Optional[str] = None

class TogglePlot3D:
    def __init__(self, data: Iterable[TogglableData]):
        self.data = data
        self.scatters: List[PathCollection] = []
        self.text_to_scatter: Dict[Text, PathCollection] = {}

        self.plot()

    def plot(self):
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': '3d'})

        for datum in self.data:
            scatter = self.ax.scatter(
                datum.X[:,0],
                datum.X[:,1],
                datum.Y,
                label=datum.name,
                alpha=datum.alpha
            )
            self.scatters.append(scatter)

        legend = self.ax.legend()
        for text, scatter in zip(legend.get_texts(), self.scatters):
            text.set_picker(True)
            self.text_to_scatter[text] = scatter

        def on_pick(event: PickEvent):
            text = event.artist
            scatter = self.text_to_scatter[text]

            visible = not scatter.get_visible()
            scatter.set_visible(visible)
            text.set_alpha(1.0 if visible else 0.2)

            self.fig.canvas.draw()

        self.fig.canvas.mpl_connect('pick_event', on_pick)

    def show(self):
        self.fig.show()
        plt.show()