import numpy as np
from matplotlib import cbook
from matplotlib.collections import PolyCollection, TriMesh
from matplotlib.colors import Normalize
from matplotlib.tri.triangulation import Triangulation
from numpy import ma

# my_cmap = [[1.0, 0.0, 0.0, 0.0] for i in np.arange(256)]
# my_cmap[-1] = [1.0, 0.0, 0.0, 1.0]
# # cmap = pl.cm.RdBu
# # my_cmap = cmap(np.arange(cmap.N))
# # my_cmap[:, -1] = 0.0
# # my_cmap[0, -1] = 1.0
# ListedColormap(my_cmap)


class CustomTriMesh(TriMesh):
    """
    Allows partially transparent color maps in TriMesh plots.
    """

    def __init__(self, triangulation, **kwargs):
        super().__init__(triangulation, **kwargs)
        # if true use self.alpha for all values and ignore cmap alpha value.
        self.override_cmap_alpha = True

    def set_override_cmap_alpha(self, override_cmap_alpha):
        self.override_cmap_alpha = override_cmap_alpha

    def to_rgba(self, x, alpha=None, bytes=False, norm=True):
        # todo check alphas...
        # First check for special case, image input:
        try:
            if x.ndim == 3:
                if x.shape[2] == 3:
                    if alpha is None:
                        alpha = 1
                    if x.dtype == np.uint8:
                        alpha = np.uint8(alpha * 255)
                    m, n = x.shape[:2]
                    xx = np.empty(shape=(m, n, 4), dtype=x.dtype)
                    xx[:, :, :3] = x
                    xx[:, :, 3] = alpha
                elif x.shape[2] == 4:
                    xx = x
                else:
                    raise ValueError("third dimension must be 3 or 4")
                if xx.dtype.kind == "f":
                    if norm and (xx.max() > 1 or xx.min() < 0):
                        raise ValueError(
                            "Floating point image RGB values "
                            "must be in the 0..1 range."
                        )
                    if bytes:
                        xx = (xx * 255).astype(np.uint8)
                elif xx.dtype == np.uint8:
                    if not bytes:
                        xx = xx.astype(np.float32) / 255
                else:
                    raise ValueError(
                        "Image RGB array must be uint8 or "
                        "floating point; found %s" % xx.dtype
                    )
                return xx
        except AttributeError:
            # e.g., x is not an ndarray; so try mapping it
            pass

        # This is the normal case, mapping a scalar array:
        x = ma.asarray(x)
        if norm:
            x = self.norm(x)
        if self.override_cmap_alpha:
            rgba = self.cmap(x, alpha=alpha, bytes=bytes)
        else:
            rgba = self.cmap(x, bytes=bytes)
        return rgba


def tripcolor_costum(
    ax,
    *args,
    alpha=1.0,
    override_cmap_alpha=True,
    norm=None,
    cmap=None,
    vmin=None,
    vmax=None,
    shading="flat",
    facecolors=None,
    **kwargs,
):
    """
    Create a pseudocolor plot of an unstructured triangular grid.

    The triangulation can be specified in one of two ways; either::

      tripcolor(triangulation, ...)

    where triangulation is a :class:`matplotlib.tri.Triangulation`
    object, or

    ::

      tripcolor(x, y, ...)
      tripcolor(x, y, triangles, ...)
      tripcolor(x, y, triangles=triangles, ...)
      tripcolor(x, y, mask=mask, ...)
      tripcolor(x, y, triangles, mask=mask, ...)

    in which case a Triangulation object will be created.  See
    :class:`~matplotlib.tri.Triangulation` for a explanation of these
    possibilities.

    The next argument must be *C*, the array of color values, either
    one per point in the triangulation if color values are defined at
    points, or one per triangle in the triangulation if color values
    are defined at triangles. If there are the same number of points
    and triangles in the triangulation it is assumed that color
    values are defined at points; to force the use of color values at
    triangles use the kwarg ``facecolors=C`` instead of just ``C``.

    *shading* may be 'flat' (the default) or 'gouraud'. If *shading*
    is 'flat' and C values are defined at points, the color values
    used for each triangle are from the mean C of the triangle's
    three points. If *shading* is 'gouraud' then color values must be
    defined at points.

    The remaining kwargs are the same as for
    :meth:`~matplotlib.axes.Axes.pcolor`.
    """
    cbook._check_in_list(["flat", "gouraud"], shading=shading)

    tri, args, kwargs = Triangulation.get_from_args_and_kwargs(*args, **kwargs)

    # C is the colors array defined at either points or faces (i.e. triangles).
    # If facecolors is None, C are defined at points.
    # If facecolors is not None, C are defined at faces.
    if facecolors is not None:
        C = facecolors
    else:
        C = np.asarray(args[0])

    # If there are a different number of points and triangles in the
    # triangulation, can omit facecolors kwarg as it is obvious from
    # length of C whether it refers to points or faces.
    # Do not do this for gouraud shading.
    if (
        facecolors is None
        and len(C) == len(tri.triangles)
        and len(C) != len(tri.x)
        and shading != "gouraud"
    ):
        facecolors = C

    # Check length of C is OK.
    if (facecolors is None and len(C) != len(tri.x)) or (
        facecolors is not None and len(C) != len(tri.triangles)
    ):
        raise ValueError(
            "Length of color values array must be the same "
            "as either the number of triangulation points "
            "or triangles"
        )

    # Handling of linewidths, shading, edgecolors and antialiased as
    # in Axes.pcolor
    linewidths = (0.25,)
    if "linewidth" in kwargs:
        kwargs["linewidths"] = kwargs.pop("linewidth")
    kwargs.setdefault("linewidths", linewidths)

    edgecolors = "none"
    if "edgecolor" in kwargs:
        kwargs["edgecolors"] = kwargs.pop("edgecolor")
    ec = kwargs.setdefault("edgecolors", edgecolors)

    if "antialiased" in kwargs:
        kwargs["antialiaseds"] = kwargs.pop("antialiased")
    if "antialiaseds" not in kwargs and ec.lower() == "none":
        kwargs["antialiaseds"] = False

    if shading == "gouraud":
        if facecolors is not None:
            raise ValueError(
                "Gouraud shading does not support the use " "of facecolors kwarg"
            )
        if len(C) != len(tri.x):
            raise ValueError(
                "For gouraud shading, the length of color "
                "values array must be the same as the "
                "number of triangulation points"
            )
        collection = CustomTriMesh(tri, **kwargs)
    else:
        # Vertices of triangles.
        maskedTris = tri.get_masked_triangles()
        verts = np.stack((tri.x[maskedTris], tri.y[maskedTris]), axis=-1)

        # Color values.
        if facecolors is None:
            # One color per triangle, the mean of the 3 vertex color values.
            C = C[maskedTris].mean(axis=1)
        elif tri.mask is not None:
            # Remove color values of masked triangles.
            C = C[~tri.mask]
        # todo: override!!!
        collection = PolyCollection(verts, **kwargs)

    collection.set_alpha(alpha)
    collection.set_override_cmap_alpha(override_cmap_alpha)
    collection.set_array(C)
    if norm is not None and not isinstance(norm, Normalize):
        raise ValueError("'norm' must be an instance of 'Normalize'")
    collection.set_cmap(cmap)
    collection.set_norm(norm)
    if vmin is not None or vmax is not None:
        collection.set_clim(vmin, vmax)
    else:
        collection.autoscale_None()
    ax.grid(False)

    minx = tri.x.min()
    maxx = tri.x.max()
    miny = tri.y.min()
    maxy = tri.y.max()
    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.add_collection(collection)
    return ax, collection
