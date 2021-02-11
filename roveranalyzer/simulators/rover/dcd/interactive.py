import json
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import QuadMesh

from roveranalyzer.simulators.rover.dcd.dcd_map import DcdMap2D


class Interactive:
    def __init__(self, ax: plt.Axes, title=""):
        self.ax = ax
        self.fig = ax.figure
        self.title = title
        self._jobs = {}

    def show(self):
        self._build_tk()
        self._register_callbacks()
        self._pack()

        tk.mainloop()

    def on_closing(self):
        self.root.quit()  # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    def _build_tk(self):
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.wm_title(self.title)
        self.main_frame = tk.Frame(self.root)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.main_frame)
        self.toolbar.update()

    def _register_callbacks(self):
        self.hover_hdl = self.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.hover_hdl = self.canvas.mpl_connect(
            "motion_notify_event", self.handle_hover
        )
        self.btn_down_hdl = self.canvas.mpl_connect(
            "button_press_event", self.handle_button_press_event
        )
        self.btn_rel_hdl = self.canvas.mpl_connect(
            "button_release_event", self.handle_button_release_event
        )

    def _wait(self, ms, job_name, callback, *args):
        # cancel previous job because of quick succession of events (wait ms)
        if job_name in self._jobs.keys() and self._jobs[job_name]:
            self.root.after_cancel(self._jobs[job_name])
        self._jobs[job_name] = self.root.after(ms, callback, *args)

    def handle_hover(self, event):
        pass

    def handle_button_press_event(self, event):
        pass

    def handle_button_release_event(self, event):
        pass

    def handle_key_press_evnet(self, event):
        pass

    def _pack(self):
        self.toolbar.pack(side=tk.TOP)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.main_frame.pack()

    def on_key_press(self, event):
        print("you pressed {}".format(event.key))
        key_press_handler(event, self.canvas, self.toolbar)


class InteractiveTableTimeNodeSlider(Interactive):
    """
    Abstract base class for interactive plot with sliders (node_id and time)
    and detail table field. Extend from this class and implement the
    update methods. The update methods are called when changes in the slider or
    mouse movements are registered. All attributes (self.[x, y, node_id, time] are
    updated beforehand and can be used from the update methods.

    :dcd: a DcDMap2D instance
    :ax: the axes / figure used for interaction.
    """

    def __init__(self, dcd: DcdMap2D, ax: plt.Axes):
        super().__init__(ax)
        self.dcd = dcd
        self.table_var = {}
        self.x = 0.0
        self.y = 0.0
        self.tracking = False
        self.info_btn_txt = tk.StringVar()

        self.time_vals = (
            self.dcd._map.index.get_level_values("simtime").unique().to_numpy()
        )
        self.id_vals = self.dcd._map.index.get_level_values("ID").unique().to_numpy()
        self.node_id = int(self.id_vals.min())
        self.time = int(self.time_vals.min())

    def _build_tk(self):
        super()._build_tk()
        self.table_frame = tk.Frame(master=self.main_frame, width=50)
        self.info_btn = tk.Button(master=self.table_frame, command=self.info_btn_cmd)
        self.info_btn_cmd()
        self.table = ScrolledText(master=self.table_frame, width=50)
        self.update_table()

        self.node_id_frame = tk.Frame(self.main_frame)
        self.node_id_lbl = tk.Label(self.node_id_frame, text="Node ID", width=7)
        self.node_id_txt = tk.Entry(self.node_id_frame, width=5, bg="white")
        self.node_id_txt.delete(0, tk.END)
        self.node_id_txt.insert(10, self.node_id)
        self.node_id_txt.bind("<Return>", self.enter)
        self.node_id_scale = tk.Scale(
            self.node_id_frame,
            from_=self.id_vals.min(),
            to=self.id_vals.max(),
            orient=tk.HORIZONTAL,
            showvalue=False,
            resolution=-1,
            command=self.cmd_node_id_scale,
        )

        self.time_frame = tk.Frame(self.main_frame)
        self.time_lbl = tk.Label(self.time_frame, text="Time", width=7)
        self.time_txt = tk.Entry(self.time_frame, width=5, bg="white")
        self.time_txt.delete(0, tk.END)
        self.time_txt.insert(10, self.time)
        self.time_txt.bind("<Return>", self.enter)
        self.time_scale = tk.Scale(
            self.time_frame,
            from_=self.time_vals.min(),
            to=self.time_vals.max(),
            orient=tk.HORIZONTAL,
            showvalue=False,
            resolution=-1,
            command=self.cmd_time_scale,
        )

    def _pack(self):
        self.toolbar.pack(side=tk.TOP)

        self.table_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.info_btn.pack(side=tk.TOP, fill=tk.X)
        self.table.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.node_id_lbl.pack(side=tk.LEFT)
        self.node_id_txt.pack(side=tk.LEFT)
        self.node_id_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.node_id_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.time_lbl.pack(side=tk.LEFT)
        self.time_txt.pack(side=tk.LEFT)
        self.time_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.time_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        self.main_frame.pack()

    def info_btn_cmd(self):
        self.tracking = not self.tracking
        if self.tracking:
            self.info_btn.configure(text="<<<<tracking>>>>")
        else:
            self.info_btn.configure(text="<<<<Start Tracking>>>>")

    def enter(self, event):
        if event.widget == self.time_txt:
            old_val = self.time_scale.get()
            try:
                val = event.widget.get().strip()
                self.time_scale.set(float(val))
                self.update_slider("time", val)
            except Exception as e:
                self.time_scale.set(old_val)
                self.time_txt.delete(0, tk.END)
                self.time_txt.insert(10, self.time)
        elif event.widget == self.node_id_txt:
            old_val = self.node_id_scale.get()
            try:
                val = event.widget.get().strip()
                self.node_id_scale.set(float(val))
                self.update_slider("node_id", val)
            except Exception as e:
                self.node_id_scale.set(old_val)
                self.node_id_txt.delete(0, tk.END)
                self.node_id_txt.insert(10, self.node_id)

    def cmd_node_id_scale(self, event):
        self._wait(5, "scale_event", self.update_slider, "node_id", event)

    def cmd_time_scale(self, event):
        self._wait(5, "scale_event", self.update_slider, "time", event)

    def handle_button_press_event(self, event):
        self.tracking = False
        self.info_btn.configure(text="<<<<Start Tracking>>>>")

    def handle_hover(self, event):
        if not (self.ax.contains(event))[0] or not self.tracking:
            return
        self._wait(5, "hover_event", self.update_cursor, event.xdata, event.ydata)

    def update_slider(self, slider, event):
        if slider == "node_id":
            _t = np.abs(self.id_vals - float(event)).argmin()
            self.node_id = self.id_vals[_t]
            self.node_id_txt.delete(0, tk.END)
            self.node_id_txt.insert(10, self.node_id)
        elif slider == "time":
            _t = np.abs(self.time_vals - float(event)).argmin()
            self.time = self.time_vals[_t]
            self.time_txt.delete(0, tk.END)
            self.time_txt.insert(10, self.time)
        else:
            raise ValueError(f"unexpected slider {slider}")

        self.update_plot()

    def update_cursor(self, x_event, y_event):

        if all([e is not None for e in [x_event, y_event]]):
            self.x = (
                np.floor(x_event / self.dcd.meta.cell_size) * self.dcd.meta.cell_size
            )
            self.y = (
                np.floor(y_event / self.dcd.meta.cell_size) * self.dcd.meta.cell_size
            )
            self.update_table()

    def update_plot(self):
        raise NotImplementedError()

    def update_table(self):
        raise NotImplementedError()


class Interactive2DDensityPlot(InteractiveTableTimeNodeSlider):

    """
    2D DensityMap plot of some scenario (x, y head map)
    """

    def __init__(self, dcd: DcdMap2D, ax: plt.Axes):
        super().__init__(dcd, ax)

        self.quadMesh = None
        for collection in ax.collections:
            if type(collection) == QuadMesh:
                self.quadMesh = collection
        if self.quadMesh is None:
            raise ValueError("expected a QuadMesh collection in passed axis.")

    def update_plot(self):
        try:
            self.dcd.update_color_mesh(self.quadMesh, self.time, self.node_id)
            self.update_table()
        except KeyError as e:
            print("key err")
            pass
        finally:
            self.canvas.draw_idle()

    def update_table(self):
        info = self.dcd.info_dict(self.x, self.y, self.time, self.node_id)
        self.table.config(state=tk.NORMAL)
        self.table.delete("1.0", tk.END)
        self.table.insert(tk.INSERT, json.dumps(info, indent=2, sort_keys=True))
