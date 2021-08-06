# # encoding:utf-8
# import glob
#
#
# # def backup_code(source_dir, target_dir):
# #     import os
# #     import fnmatch
# #     from shutil import copyfile
# #
# #     for root, dirs, files in os.walk(source_dir):
# #         # 只备份一级目录下的.py或者./deep_dialog下的.py，防止递归备份
# #         if root != source_dir and root != os.path.join(source_dir, 'deep_dialog'):
# #             continue
# #
# #         for filename in fnmatch.filter(files, '*.py'):
# #             source_file = os.path.join(root, filename)
# #             target_file = source_file.replace(source_dir, target_dir, 1)
# #             if not os.path.exists(os.path.dirname(target_file)):
# #                 os.makedirs(os.path.dirname(target_file))
# #             copyfile(source_file, target_file)
#
#
# backup_code('.', './hello')
