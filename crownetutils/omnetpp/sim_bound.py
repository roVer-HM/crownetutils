class SimBound:
    """Offset and simulation bounding box of the simulation.

    The **offset** defines the translation of point `0` compared
    to the origin. In case of geo located data the origin of the
    used coordinate system. Inside the simulation the point `0`
    has the coordinate (0, 0)

    The `sim_box` is defined as the point `S`, relative to the
    simulation origin `0`, and a width and height.


         +------------------------+
         |                        |
         |   +----------+         |
         |   |          |         |
      ^  |   S----------+         |
      |  |                        |
      y  0------------------------+
         x -->

    """

    def __init__(self, offset, sim_bbox) -> None:
        self.offset = offset
        self.sim_bbox = sim_bbox
        self.abs_bound = self.sim_bbox[1] - self.sim_bbox[0]

    @property
    def sim_width(self):
        return self.abs_bound[0]

    @property
    def sim_height(self):
        return self.abs_bound[1]

    @property
    def sim_offset(self):
        return -1 * self.sim_bbox[0]
