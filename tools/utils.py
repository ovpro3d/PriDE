import os
import glob
import shutil
import logging

def backup_code(work_dir, verbose=False):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 使用 '**/*' 确保递归匹配所有文件及子目录
    patterns = [
        'projects/**/**', 'tools/**/**'
    ]

    for pattern in patterns:
        # 使用 recursive=True 来递归查找文件
        for file in glob.glob(pattern, recursive=True):
            src = os.path.join(base_dir, file)  # 源文件路径
            if not os.path.isfile(src):
                continue

            file_extension = os.path.splitext(src)[-1]
            if not file_extension in ['.py', '.cpp', '.cu', '.h']:
                continue

            dst = os.path.join(work_dir, 'backup', os.path.dirname(file))  # 目标路径，保留目录结构

            if verbose:
                logging.info('Copying %s -> %s' % (os.path.relpath(src), os.path.relpath(dst)))

            # 确保目标文件夹存在
            os.makedirs(dst, exist_ok=True)

            # 拷贝文件，包括元数据
            shutil.copy2(src, dst)
