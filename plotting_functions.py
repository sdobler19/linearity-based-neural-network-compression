import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})


def weight_plot(plotting_data, title = "", path_to_save = ""):
    plt.plot(plotting_data["compression_thresholds"], plotting_data["n_weights_lay"], label = "n layer weights")
    plt.plot(plotting_data["compression_thresholds"], plotting_data["n_weights_skip"], label = "n skip weights")
    plt.plot(plotting_data["compression_thresholds"], plotting_data["n_weights"], label = "n weights")
    plt.gca().invert_xaxis()
    plt.xlabel("Activation rate - Threshold")
    plt.ylabel("Number of weights")
    plt.legend()

    plt.style.use("bmh")

    if path_to_save:
        plt.savefig(path_to_save, bbox_inches='tight')
    else:
        plt.show()


def loss_plot(plotting_data, title = "", path_to_save = ""):
    plotting_data.plot.line(x = "compression_thresholds", y = "Loss")
    plt.gca().invert_xaxis()
    plt.xlabel("Activation rate - Threshold")
    plt.ylabel("Accuracy")

    plt.style.use("bmh")
    plt.legend().remove()

    if path_to_save:
        plt.savefig(path_to_save, bbox_inches='tight')
    else:
        plt.show()



def weight_loss_plot(plotting_data, title = "", path_to_save = ""):
    plt.plot(plotting_data["n_weights"], plotting_data["Loss"])
    plt.xlabel("Number of weights")
    plt.ylabel("Accuracy")
    plt.gca().invert_xaxis()

    if path_to_save:
        plt.savefig(path_to_save, bbox_inches='tight')
    else:
        plt.show()