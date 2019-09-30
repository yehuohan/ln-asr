# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# 设置源码路径（use absolute path）
import os
import sys
sys.path.insert(0, os.path.abspath('../../lnasr'))


# -- Project information -----------------------------------------------------

# 项目信息
project = 'ln-asr'
copyright = '2019, yehuohan@gmail.com'
author = 'yehuohan@gmail.com'


# -- General configuration ---------------------------------------------------

# Sphinx扩展模块
extensions = [
    'sphinx.ext.autodoc',
    'm2r',
]

# 模版路径
templates_path = ['_templates']

# 文档语言
language = 'zh'

# 忽略的文件和目录
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# 文档主题样式
html_theme = 'sphinxdoc'

# HTML静态文件(style sheets ...)
html_static_path = ['_static']

# 文档支持格式
source_suffix = {
    '.rst' : 'restructuredtext',
    '.md' : 'markdown',
}
