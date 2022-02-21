import matplotlib.pyplot as plt


def show_image_with_colorbar(img, ax, shrink=0.4, pad=0.01, labelsize=3, vmin=None, vmax=None):
    if vmin :
        im = ax.imshow(img, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(img)
    cb = plt.colorbar(im, ax=ax, shrink=shrink, pad=pad, orientation="horizontal")
    cb.ax.tick_params(labelsize=labelsize, rotation=45)
