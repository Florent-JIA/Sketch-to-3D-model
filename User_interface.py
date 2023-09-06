import os
import subprocess
import tkinter as tk
import tkinter.font as tkFont
from multiprocessing import Process
from tkinter import filedialog, Text, messagebox
from PIL import Image, ImageTk
from global_test import sketch2model
from voxel_visualization.visualize import visualizer
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class SoftwareInterfaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SeAD")

        # 设置窗口的默认大小
        window_width = 1600
        window_height = 1000
        self.root.geometry(f"{window_width}x{window_height}")

        # 计算居中的位置
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        center_x = int((screen_width - window_width) / 2)
        center_y = int((screen_height - window_height) / 2)

        # 将窗口移动到计算出的居中位置
        self.root.geometry(f"+{center_x}+{center_y}")

        # 创建标题栏
        title_frame = tk.Frame(root, bg="darkblue", width=root.winfo_screenwidth(), height=30)
        title_frame.pack()

        # 在标题栏中添加 Segula Aided Design 文字
        title_label = tk.Label(title_frame, text="Segula Aided Design", bg="darkblue", fg="white", font=tkFont.Font(family="Helvetica", size=20))
        title_label.pack()

        # 创建水平排列的方框：Import sketch，2D Generator & Output，Start conversion
        horizontal_frame = tk.Frame(root, bg="white")
        horizontal_frame.pack(pady=10)

        # Image import frame
        self.import_sketch_frame = tk.Frame(horizontal_frame, width=400, height=400, bd=2, relief="solid")
        self.import_sketch_frame.pack(side="left", padx=50, pady=10)

        # Vertical frame to hold 2D generator and output directory frames
        vertical_frame = tk.Frame(horizontal_frame, bg="white")
        vertical_frame.pack(side="left", padx=50)

        self.category_frame = tk.Frame(vertical_frame, width=50, height=50, bd=2, relief="solid")
        self.category_frame.pack(side="top", padx=10, pady=10)
        self.create_category_menu()

        # 2D generator frame inside the vertical frame
        # self.generator_frame = tk.Frame(vertical_frame, width=50, height=200, bd=2, relief="solid")
        # self.generator_frame.pack(side="top", padx=10, pady=10)

        # Output directory frame inside the vertical frame
        self.outputdir_frame = tk.Frame(vertical_frame, width=50, height=200, bd=2, relief="solid")
        self.outputdir_frame.pack(side="top", padx=10, pady=10)

        # Thresh frame below the outputdir_frame
        self.thresh_frame = tk.Frame(vertical_frame, bd=2, relief="solid")
        self.thresh_frame.pack(side="top", padx=10, pady=10)
        thresh_label = tk.Label(self.thresh_frame, text="thresh for silhouette: ")
        thresh_label.pack(side="left")
        self.thresh_entry = tk.Entry(self.thresh_frame)
        self.thresh_entry.pack(side="left")
        self.thresh_entry.insert(0, "0.95")

        # Conversion button below the thresh_frame
        conversion_button = tk.Button(vertical_frame, text="conversion", command=self.start_conversion)
        conversion_button.pack(side="top", padx=10, pady=10)

        # Output frame
        self.output_frame = tk.Frame(horizontal_frame, bg="white", width=300, height=400, bd=2, relief="solid")
        self.output_frame.pack(side="left", padx=50, pady=10)
        self.output_text = Text(self.output_frame, wrap=tk.WORD, height=20, width=40)
        self.output_text.insert(tk.END, "Conversion result")
        self.output_text.pack(pady=10, padx=10)

        # 新增的代码
        self.pointcloud_frame = tk.Frame(root, bg="white", bd=2, relief="solid")
        self.pointcloud_frame.pack(side="bottom", padx=10, pady=10, fill="both", expand=True)
        # 在pointcloud_frame中添加标题
        title_label = tk.Label(self.pointcloud_frame, text="result of 3D conversion", anchor="n")
        title_label.pack(fill="both", pady=5)

        # Initial buttons
        self.import_button = tk.Button(self.import_sketch_frame, text="Import sketch", command=self.import_image)
        self.import_button.pack(pady=80)

        # self.choose_generator_button = tk.Button(self.generator_frame, text="Choose your 2D generator", command=self.choose_generator)
        # self.choose_generator_button.pack()

        self.choose_output_folder_button = tk.Button(self.outputdir_frame, text="Choose your output folder", command=self.choose_output_folder)
        self.choose_output_folder_button.pack()

        self.open3D_button = tk.Button(self.pointcloud_frame, text="open 3D model", command=self.open_3d_model)
        self.open3D_button.pack(pady=20)

        # Initialize other UI components to None
        self.image_label = None
        self.path_label = None
        self.generator_path_label = None
        self.outputdir_path_label = None
        self.change_button = None
        self.full_image_path = None
        self.full_generator_path = None
        self.full_outputdir_path = None

        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.category_path = os.path.join(self.base_path, "Reconstruction_2D", "parameters")

    def create_category_menu(self):
        self.category_button = tk.Button(self.category_frame, text="Choose the category",
                                         command=self.show_category_dropdown)
        self.category_button.pack()

        # 创建下拉菜单，但不立即显示它
        self.dropdown_menu = tk.Menu(self.category_frame, tearoff=0)
        categories = ["Car", "Plane", "Chair", "Other"]
        for category in categories:
            self.dropdown_menu.add_command(label=category, command=lambda c=category: self.set_category(c))

        # 初始化category_label
        self.category_label = None


    def show_category_dropdown(self):
        # 展示下拉菜单
        self.dropdown_menu.post(self.category_button.winfo_rootx(),
                                self.category_button.winfo_rooty() + self.category_button.winfo_height())

    def set_category(self, category):
        # 设置所选类别并更新显示路径
        self.selected_category = category

        # 获取UI程序的母文件夹路径

        self.category_path = os.path.join(self.base_path, "Reconstruction_2D", "parameters", category)
        self.full_generator_path = os.path.join(self.category_path, '100st_epoch_G_net.pth')

        # 删除旧的路径标签（如果存在）
        if self.category_label:
            self.category_label.destroy()

        # 显示新的路径
        self.display_truncated_path(self.category_path, 'category')

        # 隐藏下拉菜单
        self.dropdown_menu.unpost()

    def start_conversion(self):
        test_sketch_path = self.full_image_path
        netG_weights_path = self.full_generator_path
        output_dir = self.full_outputdir_path
        thresh_silhouette = float(self.thresh_entry.get())

        # Verify all parameters are set
        if not test_sketch_path or not netG_weights_path or not output_dir or not thresh_silhouette:
            messagebox.showerror("Error", "All parameters are not set!")
            return

        # Call conversion_test function
        sub_output_folder_path = sketch2model(test_sketch_path, netG_weights_path, output_dir, float(thresh_silhouette))
        self.sub_output_folder_path = sub_output_folder_path
        self.show_output_buttons(sub_output_folder_path)

    def import_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        if file_path:
            self.full_image_path = file_path  # Save the full path
            # Clear existing UI components for import_sketch_frame
            if self.image_label:
                self.image_label.destroy()
                self.image_label = None
            if self.path_label:
                self.path_label.destroy()
                self.path_label = None
            if self.change_button:
                self.change_button.destroy()
                self.change_button = None
            if self.import_button:
                self.import_button.destroy()
                self.import_button = None
            # Display the truncated file path
            self.display_truncated_path(file_path, 'image')
            # Load and display the image
            image = Image.open(file_path).resize((256, 256), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            self.image_label = tk.Label(self.import_sketch_frame, image=photo)
            self.image_label.image = photo
            self.image_label.pack(fill="both", expand="yes")
            # Add the "change your sketch" button below the image
            self.change_button = tk.Button(self.import_sketch_frame, text="Change your sketch",
                                           command=self.import_image)
            self.change_button.pack(pady=5)

    # def choose_generator(self):
    #     file_path = filedialog.askopenfilename(initialdir=self.category_path, filetypes=[("PTH files", "*.pth")])
    #     if file_path:
    #         self.full_generator_path = file_path  # Save the full path
    #         # Clear existing UI components for generator_frame
    #         if self.generator_path_label:
    #             self.generator_path_label.destroy()
    #             self.generator_path_label = None
    #         if self.choose_generator_button:
    #             self.choose_generator_button.destroy()
    #             self.choose_generator_button = None
    #         # Display the truncated file path
    #         self.display_truncated_path(file_path, 'generator')
    #         # Restore the button for future selections
    #         self.choose_generator_button = tk.Button(self.generator_frame, text="Choose your 2D generator",
    #                                                  command=self.choose_generator)
    #         self.choose_generator_button.pack()

    def choose_output_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.full_outputdir_path = folder_path  # Save the full path
            # Clear existing UI components for outputdir_frame
            if self.outputdir_path_label:
                self.outputdir_path_label.destroy()
                self.outputdir_path_label = None
            if self.choose_output_folder_button:
                self.choose_output_folder_button.destroy()
                self.choose_output_folder_button = None
            # Display the truncated folder path
            self.display_truncated_path(folder_path, 'outputdir')
            # Restore the button for future selections
            self.choose_output_folder_button = tk.Button(self.outputdir_frame, text="Choose your output folder",
                                                         command=self.choose_output_folder)
            self.choose_output_folder_button.pack()

    def show_output_buttons(self, sub_output_folder_path):
        # 清空output_frame中的现有按钮或组件
        for widget in self.output_frame.winfo_children():
            widget.destroy()

        # 生成按钮
        btn_open_folder = tk.Button(self.output_frame, text="open output folder", command=lambda: self.open_folder(sub_output_folder_path))
        btn_real_picture = tk.Button(self.output_frame, text="real picture", command=lambda: self.open_image(os.path.join(sub_output_folder_path, "generated_rgb.png")))
        btn_silhouette = tk.Button(self.output_frame, text="silhouette", command=lambda: self.open_image(os.path.join(sub_output_folder_path, "silhouette.png")))
        btn_depth_map = tk.Button(self.output_frame, text="depth map", command=lambda: self.open_image(os.path.join(sub_output_folder_path, 'batch0000', 'depth_image.png')))
        btn_partial_spherical = tk.Button(self.output_frame, text="particial spherical rendering", command=lambda: self.open_image(os.path.join(sub_output_folder_path, 'batch0000', 'sph_partial.png')))
        btn_full_spherical = tk.Button(self.output_frame, text="full spherical rendering", command=lambda: self.open_image(os.path.join(sub_output_folder_path, 'batch0000', 'sph_full.png')))
        btn_depth_projection = tk.Button(self.output_frame, text="depth projection", command=lambda: self.open_npy(os.path.join(sub_output_folder_path, 'batch0000', 'proj_depth.npy')))
        btn_spherical_projection = tk.Button(self.output_frame, text="spherical rendering projection", command=lambda: self.open_npy(os.path.join(sub_output_folder_path, 'batch0000', 'pred_proj_sph.npy')))

        # 放置按钮
        btn_open_folder.pack(pady=5)
        btn_real_picture.pack(pady=5)
        btn_silhouette.pack(pady=5)
        btn_depth_map.pack(pady=5)
        btn_partial_spherical.pack(pady=5)
        btn_full_spherical.pack(pady=5)
        btn_depth_projection.pack(pady=5)
        btn_spherical_projection.pack(pady=5)

    def open_folder(self, folder_path):
        subprocess.run(["xdg-open", folder_path])

    def open_image(self, image_path):
        subprocess.run(["xdg-open", image_path])

    def open_npy(self, npy_path):
        process = Process(target=visualizer, args=(npy_path,))
        process.start()

    def open_3d_model(self):
        # 使用selenium控制Firefox浏览器
        driver = webdriver.Firefox(executable_path='/home/oem/Downloads/code/geckodriver')  # 提供geckodriver的路径
        # 设定一个特定的窗口大小
        desired_width = 1500
        desired_height = 524
        driver.set_window_size(desired_width, desired_height)

        # 计算居中的位置
        screen_width = driver.execute_script("return screen.width;")
        screen_height = driver.execute_script("return screen.height;")
        x_position = (screen_width - desired_width) / 2
        y_position = (screen_height - desired_height) / 2 + 220

        # 将窗口移动到计算的位置
        driver.set_window_position(x_position, y_position)

        driver.get("https://3dviewer.net/#model=assets/models/solids.obj,assets/models/solids.mtl")

        # 设置一个等待
        wait = WebDriverWait(driver, 10)

        # 使用WebDriverWait来确保按钮完全可见并可以点击
        open_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//i[contains(@class, 'icon-open')]")))

        # 如果按钮还被其他元素遮挡，可以尝试使用ActionChains来模拟鼠标移动到按钮上并点击
        action = ActionChains(driver)
        action.move_to_element(open_button).click().perform()

        # 等待文件选择框弹出
        time.sleep(2)

        try:
            accept_button = wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//div[@class='ov_button ov_floating_panel_button' and text()='Accept']")))
            accept_button.click()
        except:
            # 如果按钮不存在或不可点击，这里将捕获异常，您可以添加更多的处理代码或者仅仅忽略
            pass

        # 使用selenium模拟文件选择操作
        file_input = driver.find_element(By.XPATH, "//input[@type='file']")
        # 提供需要上传的文件的完整路径
        file_input.send_keys(os.path.join(self.sub_output_folder_path, 'batch0000', '0000_12_pred_voxel.obj'))



        # 如果文件选择框没有自动关闭，尝试点击页面上的其他元素关闭它
        driver.find_element(By.XPATH, "//body").click()
        driver.find_element(By.XPATH, "//body").send_keys(Keys.ESCAPE)

    def display_truncated_path(self, path, label_type):
        display_path = path
        if len(path) > 30:
            display_path = '...' + path[-27:]
        if label_type == 'image':
            self.path_label = tk.Label(self.import_sketch_frame, text=display_path, anchor="w")
            self.path_label.pack(fill="both", pady=5)
        elif label_type == 'category':
            self.category_label = tk.Label(self.category_frame, text=display_path, anchor="w")
            self.category_label.pack(fill="both", pady=5)
        elif label_type == 'generator':
            self.generator_path_label = tk.Label(self.generator_frame, text=display_path, anchor="w")
            self.generator_path_label.pack(fill="both", pady=5)
        elif label_type == 'outputdir':
            self.outputdir_path_label = tk.Label(self.outputdir_frame, text=display_path, anchor="w")
            self.outputdir_path_label.pack(fill="both", pady=5)



if __name__ == "__main__":
    root = tk.Tk()
    app = SoftwareInterfaceApp(root)
    root.mainloop()













































