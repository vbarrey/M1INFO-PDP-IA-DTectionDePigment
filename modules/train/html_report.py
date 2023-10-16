import os
import matplotlib.pyplot 
class HTMLReport(object):
    def __init__(self,
    date,
    dataset_path,
    nb_epochs,
    batch_size,
    img_size,
    train_size,
    test_size,
    total_time,
    loss_graph,
    conf_matrix,
    precision,
    recall,
    example_imgs):
        self.loss_graph = loss_graph
        self.conf_matrix = conf_matrix
        self.example_imgs = example_imgs
        self.html = f"<h1>Training Report {date}</h1>"
        self.html += "<h2>Parameters</h2>\n"
        self.html += f"<p>Dataset : {dataset_path}</p>\n"
        self.html += f"<p>Epochs : {nb_epochs}</p>\n"
        self.html += f"<p>Batch Size : {batch_size}</p>\n"
        self.html += f"<p>Image Size : {img_size}</p>\n"
        self.html += f"<p>Size of training dataset : {train_size}</p>\n"
        self.html += f"<p>Size of test dataset : {test_size}</p>\n"
        self.html += f"<p>Training duration: {total_time} min</p>\n"
        self.html += "<h2>Training summary</h2>\n"
        self.html += "<p>Loss evolution :</p>\n"
        self.html += "<p><img src=\"./img/loss_graph.png\" /></p>\n"
        self.html += "<h2>Confusion Matrix</h2>\n"
        self.html += "<p><img src=\"./img/conf_matrix.png\" /></p>\n"
        self.html += "<h2>Precision and Recall</h2>\n"
        self.html += f"<p>Precision : {precision}</p>\n"
        self.html += f"<p>Recall : {recall}</p>\n"
        self.html += "<h2>Examples</h2>\n"
        for i in range(len(example_imgs)) :
            self.html += f"<p><img src=\"./img/example{i}.png\" /></p>\n"

    def save(self, ouputdir):
        report_dir = ouputdir+os.sep+"report"
        if not os.path.exists(report_dir) :
            os.mkdir(report_dir)

        img_dir = report_dir+os.sep+"img"
        if not os.path.exists(img_dir) :
            os.mkdir(img_dir)

        self.loss_graph.savefig(img_dir + os.sep + "loss_graph.png", bbox_inches='tight')

        self.conf_matrix.savefig(img_dir + os.sep + "conf_matrix.png", bbox_inches='tight')

        for i in range(len(self.example_imgs)) :
            self.example_imgs[i].savefig(img_dir + os.sep + f"example{i}.png", bbox_inches='tight')

        with open(report_dir+os.sep+"report.html", "w") as report_file:
            report_file.write(self.html)