import json
import tkinter as tk
from tkinter import messagebox
from tkinter.commondialog import Dialog
from tkinter.filedialog import asksaveasfilename
from tkinter.scrolledtext import ScrolledText

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import QuadMesh

from roveranalyzer.simulators.crownet.dcd.dcd_map import DcdMap2D


class MyDialog(Dialog):
    def __init__(self, root, fields, **options):
        super().__init__(**options)
        self.root = root
        self.top = tk.Toplevel(self.root)

        self.main = tk.Frame(self.top, borderwidth=4, relief="ridge")
        self.main.pack(fill=tk.BOTH, expand=True)

        self.items_gui = {}
        self.data = {}

        for key, item in fields.items():
            item_f = tk.Frame(self.main)
            lbl = tk.Label(item_f, text=item[0])
            entry = tk.Entry(item_f)
            lbl.pack(side=tk.LEFT)
            entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            item_f.pack(side=tk.TOP, fill=tk.X, expand=True)
            self.items_gui.setdefault(key, [item_f, lbl, entry])

        self.btn_f = tk.Frame(self.main)
        self.btn_ok = tk.Button(self.btn_f, text="OK", command=self.ok)
        self.btn_cancel = tk.Button(self.btn_f, text="Cancel", command=self.cancel)
        self.btn_ok.pack(side=tk.LEFT)
        self.btn_cancel.pack(side=tk.RIGHT)
        self.btn_f.pack(side=tk.BOTTOM, fill=tk.X, expand=True)
        self.top.wait_window()

    def ok(self):
        for key, ui in self.items_gui.items():
            entry = ui[2]
            self.data[key] = entry.get().strip()
        self.top.destroy()

    def cancel(self):
        self.data = None
        self.top.destroy()


class IntScale(tk.Scale):
    """
    Scale based on data array independent on actual values.
    The scale gui will use the index of the array 0...len(data) as the scale.
    For some reason 'from', 'to' only works if diff is greater than 100 (for integer)
    This IntScale uses a fixed scale of 0-128 for data input where len(data) is smaller than 100
    and then interpolates the index of the data array when the slider is moved.
    The get() and set() methods return or set *actual* values present in the underling data. Use
    get_scale(), set_scale() to set/get the slider position.

    If the slider is set using user values (form the data array) the closest value will be used e.g.:
    data = [2, 4, 5, 6, 8]
    set(4,9) --> will set slider at data[2] --> 5
    """

    def __init__(self, *arg, **kwargs):
        self.data = kwargs.pop("data")
        self.idx_max = len(self.data)

        if len(self.data) < 100:
            # gui does not work if to-from_ is smaller than 100 ?
            self.s_min = 0
            self.s_max = 128
        else:
            self.s_min = 0
            self.s_max = len(self.data)

        # underling scale used in gui
        kwargs["from_"] = self.s_min
        kwargs["to"] = self.s_max
        super().__init__(*arg, **kwargs)

    def get_idx(self, scale_value):
        r = int(np.round(scale_value * self.idx_max / self.s_max))
        return r

    def get_scale_from_idx(self, idx):
        r = int(np.round(idx * self.s_max / self.idx_max))
        return r

    def get_scale_from_value(self, value):
        _t = np.abs(self.data - float(value)).argmin()
        return self.get_scale_from_idx(_t)

    def get_scale(self):
        return super().get()

    def get_for(self, scale):
        return

    def get(self):
        """
        return user value
        """
        idx = self.get_idx(super().get())
        idx = np.min([idx, len(self.data) - 1])
        return self.data[idx]

    def set_scale(self, scale):
        super().set(scale)

    def set(self, value):
        """
        expect user value!
        """
        super().set(self.get_scale_from_value(float(value)))


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

        self.update_all()

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

    def set_ax_attr(self):
        def lim(ax, attr, val):
            vals = [float(v.strip()) for v in val.strip().split(",")]
            getattr(ax, f"set_{attr}", None)(vals[0], vals[1])

        attr_cmd = lambda ax, attr, val: ax.update({attr: val})
        items = {
            "title": ["Title", attr_cmd],
            "titlefont": [
                "Title font size",
                lambda ax, attr, val: ax.set_title(
                    ax.get_title(), fontdict={"fontsize": val}
                ),
            ],
            "xlabel": ["Label X", attr_cmd],
            "ylabel": ["Label Y", attr_cmd],
            "xlim": ["xlim", lim],
            "ylim": ["ylim", lim],
        }
        dialog = MyDialog(self.root, items)
        print(dialog.data)
        if dialog.data is not None:
            for key, val in dialog.data.items():
                if val != "":
                    cmd = items[key][1]
                    cmd(self.ax, key, val)
        self.canvas.draw_idle()

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

    def update_all(self):
        raise NotImplementedError()


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

        self.time_vals = self.dcd.valid_times()
        self.id_vals = self.dcd.all_ids(with_ground_truth=True)
        self.node_id = int(self.id_vals.min())
        self.time = int(self.time_vals.min())

    def _build_tk(self):
        super()._build_tk()
        self.table_frame = tk.Frame(master=self.main_frame, width=50)
        self.info_btn = tk.Button(master=self.table_frame, command=self.info_btn_cmd)
        self.cfg_btn = tk.Button(
            master=self.table_frame, text="Config", command=self.set_ax_attr
        )
        self.animate_btn = tk.Button(
            master=self.table_frame, text="Animate", command=self.animate_btn_cmd
        )
        self.info_btn_cmd()
        self.table = ScrolledText(master=self.table_frame, width=50)
        self.update_table()

        self.node_id_frame = tk.Frame(self.main_frame)
        self.node_id_lbl = tk.Label(self.node_id_frame, text="Node ID", width=7)
        self.node_id_txt = tk.Entry(self.node_id_frame, width=5, bg="white")
        self.node_id_txt.delete(0, tk.END)
        self.node_id_txt.insert(10, self.node_id)
        self.node_id_txt.bind("<Return>", self.enter)
        self.node_id_scale = IntScale(
            self.node_id_frame,
            data=self.id_vals,
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
        self.time_scale = IntScale(
            self.time_frame,
            data=self.time_vals,
            orient=tk.HORIZONTAL,
            showvalue=False,
            resolution=-1,
            command=self.cmd_time_scale,
        )

    def _pack(self):
        self.toolbar.pack(side=tk.TOP)

        self.table_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.info_btn.pack(side=tk.TOP, fill=tk.X)
        self.cfg_btn.pack(side=tk.TOP, fill=tk.X)
        self.animate_btn.pack(side=tk.TOP, fill=tk.X)
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

    def animate_btn_cmd(self):
        # dialog = MyDialog(self.root, {
        #     "node_id": self.node_id,
        #     "time_from": self.time_vals[0],
        #     "time_to": self.time_vals[-1]
        # })
        # if dialog.data is not None:
        file = asksaveasfilename()
        if file is None:
            return

        anim = animation.FuncAnimation(
            fig=self.fig, func=self.animate, frames=self.time_vals, blit=False
        )
        anim.save(file)

    def animate(self, frame):
        raise NotImplementedError()

    def info_btn_cmd(self):
        self.tracking = not self.tracking
        if self.tracking:
            self.info_btn.configure(text="<<<<tracking>>>>")
        else:
            self.info_btn.configure(text="<<<<Start Tracking>>>>")

    def enter(self, event):
        if event.widget == self.time_txt:
            old_val = self.time_scale.get()  # user value (not index!)
            try:
                val = event.widget.get().strip()
                self.time_scale.set(val)
                # update slider expects scale value of Scale class
                self.update_slider("time", self.time_scale.get_scale_from_value(val))
            except Exception:
                self.time_scale.set(old_val)
                self.time_txt.delete(0, tk.END)
                self.time_txt.insert(10, self.time)
        elif event.widget == self.node_id_txt:
            old_val = self.node_id_scale.get()  # user value (not index!)
            try:
                val = event.widget.get().strip()
                self.node_id_scale.set(val)
                # update slider expects scale value of Scale class
                self.update_slider(
                    "node_id", self.node_id_scale.get_scale_from_value(val)
                )
            except Exception:
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
            # _t = self.node_id_scale.get(int(event))
            _t = self.node_id_scale.get()
            self.node_id = _t
            self.node_id_txt.delete(0, tk.END)
            self.node_id_txt.insert(10, self.node_id)
        elif slider == "time":
            # _t = self.time_scale.get(int(event))
            _t = self.time_scale.get()
            self.time = _t
            self.time_txt.delete(0, tk.END)
            self.time_txt.insert(10, self.time)
        else:
            raise ValueError(f"unexpected slider {slider}")

        self.update_plot()

    def update_cursor(self, x_event, y_event):

        if all([e is not None for e in [x_event, y_event]]):
            self.x = (
                np.floor(x_event / self.dcd.metadata.cell_size)
                * self.dcd.metadata.cell_size
            )
            self.y = (
                np.floor(y_event / self.dcd.metadata.cell_size)
                * self.dcd.metadata.cell_size
            )
            self.update_table()

    def update_all(self):
        self.update_plot()
        self.update_table()

    def update_plot(self):
        raise NotImplementedError()

    def update_table(self):
        raise NotImplementedError()


class InteractiveAreaPlot(InteractiveTableTimeNodeSlider):
    """
    2D DensityMap plot of some scenario (x, y head map)
    """

    def __init__(self, dcd: DcdMap2D, ax: plt.Axes, value: str):
        super().__init__(dcd, ax)

        self.quadMesh = None
        self.value = value
        for collection in ax.collections:
            if type(collection) == QuadMesh:
                self.quadMesh = collection
        if self.quadMesh is None:
            raise ValueError("expected a QuadMesh collection in passed axis.")

    def animate(self, frame):
        print(frame)
        self.dcd.update_color_mesh(self.quadMesh, frame, self.node_id, self.value)

    def update_plot(self):
        try:
            self.dcd.update_color_mesh(
                self.quadMesh, self.time, self.node_id, self.value
            )
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


class InteractiveValueOverDistance(InteractiveTableTimeNodeSlider):
    """
    2D DensityMap plot of some scenario (x, y head map)
    """

    def __init__(
        self, dcd: DcdMap2D, ax: plt.Axes, value, update_f, update_f_args=None
    ):
        super().__init__(dcd, ax)

        self.node_id = self.id_vals[1]
        self.line = ax.lines[0]
        self.value = value
        self.update_f = update_f
        self.update_f_args = {} if update_f_args is None else update_f_args
        self.data = self.update_f(
            self.time, self.node_id, self.value, line=self.line, **self.update_f_args
        )

    def animate(self, frame):
        print(frame)
        self.update_f(
            frame, self.node_id, self.value, line=self.line, **self.update_f_args
        )
        # self.dcd.update_delay_over_distance(
        #     frame, self.node_id, "measurement_age", line=self.line
        # )

    def update_plot(self):
        try:
            self.data = self.update_f(
                self.time,
                self.node_id,
                self.value,
                line=self.line,
                **self.update_f_args,
            )

            # self.data = self.dcd.update_delay_over_distance(
            #     self.time, self.node_id, "measurement_age", line=self.line
            # )
            self.update_table()
        except KeyError as e:
            print("key err")
            pass
        finally:
            self.canvas.draw_idle()

    def update_table(self):
        info = {}
        info.setdefault("number_points", self.data.shape[0])
        info.update(self.data.describe().to_dict())
        self.table.config(state=tk.NORMAL)
        self.table.delete("1.0", tk.END)
        self.table.insert(tk.INSERT, json.dumps(info, indent=2, sort_keys=True))
