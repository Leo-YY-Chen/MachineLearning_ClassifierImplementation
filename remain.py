'''def save_dict(filename, dict):
    with open(filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=dict.keys())
        writer.writeheader()
        writer.writerow(dict)

def save_plot(filename, train_list, valid_list, plotname='accuracy'):
    fig, ax = plt.subplots(figsize=(8,4))
    plt.title(plotname)
    plt.plot(train_list, label='train '+plotname)
    plt.plot(valid_list, label='valid '+plotname, linestyle='--')
    plt.legend()
    plt.savefig(filename)
    plt.show()

def save_bar_chart(filename, label_list, data_list, plotname='Feature Importance'):
    plt.title(plotname)
    plt.bar(label_list, data_list)
    plt.xticks(rotation=30, ha='right')
    plt.savefig(filename)
    plt.show()'''