import matplotlib.pyplot as plt


def display_images(r, g, b, im_out):
    """Creates a single large figure with 4 subplots"""

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].imshow(r, cmap="gray")
    axs[0, 0].set_title("r image")

    axs[0, 1].imshow(g, cmap="gray")
    axs[0, 1].set_title("g image")

    axs[1, 0].imshow(b, cmap="gray")
    axs[1, 0].set_title("b image")

    axs[1, 1].imshow(im_out)
    axs[1, 1].set_title("Colorized image")

    # Add color bars for the grayscale images
    cbar_r = plt.colorbar(
        axs[0, 0].imshow(r, cmap="gray"), ax=axs[0, 0], orientation="vertical"
    )
    cbar_g = plt.colorbar(
        axs[0, 1].imshow(g, cmap="gray"), ax=axs[0, 1], orientation="vertical"
    )
    cbar_b = plt.colorbar(
        axs[1, 0].imshow(b, cmap="gray"), ax=axs[1, 0], orientation="vertical"
    )

    plt.tight_layout()

    plt.show()


# # Debug the indivudidual images
# plt.imshow(r, cmap="gray")
# plt.title("r image")
# plt.colorbar()
# plt.show()
# plt.imshow(g, cmap="gray")
# plt.title("g image")
# plt.colorbar()
# plt.show()
# plt.imshow(b, cmap="gray")
# plt.title("b image")
# plt.colorbar()
# plt.show()


# # Save and display image
# plt.imshow(im_out)
# plt.title("Colorized image")
# plt.show()
