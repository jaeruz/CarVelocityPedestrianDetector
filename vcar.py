#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.26
#  in conjunction with Tcl version 8.6
#    Jan 24, 2020 01:26:24 PM CST  platform: Windows NT

import sys
from deltaM import vproc

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import vcar_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    top = Toplevel1 (root)
    vcar_support.init(root, top)
    root.mainloop()

w = None
def create_Toplevel1(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel (root)
    top = Toplevel1 (w)
    vcar_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

def clicked(v):
    vproc(v)
    root.destroy()

class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        font14 = "-family Consolas -size 16 -weight normal -slant "  \
            "roman -underline 0 -overstrike 0"
        font15 = "-family Consolas -size 14 -weight normal -slant "  \
            "roman -underline 0 -overstrike 0"

        top.geometry("751x438+461+204")
        top.minsize(148, 1)
        top.maxsize(1924, 1055)
        top.resizable(1, 1)
        top.title("New Toplevel")
        top.configure(background="#008080")

        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.0, rely=0.0, relheight=1.016, relwidth=0.087)
        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#008080")

        self.Frame2 = tk.Frame(top)
        self.Frame2.place(relx=0.08, rely=0.0, relheight=0.993, relwidth=0.925)
        self.Frame2.configure(relief='groove')
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief="groove")
        self.Frame2.configure(background="#58abab")

        self.Entry1 = tk.Entry(self.Frame2)
        self.Entry1.place(relx=0.388, rely=0.276,height=44, relwidth=0.437)
        self.Entry1.configure(background="white")
        self.Entry1.configure(disabledforeground="#a3a3a3")
        self.Entry1.configure(font="-family {Courier New} -size 10")
        self.Entry1.configure(foreground="#000000")
        self.Entry1.configure(insertbackground="black")

        self.Label1 = tk.Label(self.Frame2)
        self.Label1.place(relx=0.043, rely=0.276, height=44, width=228)
        self.Label1.configure(activebackground="#58abab")
        self.Label1.configure(background="#58abab")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font=font14)
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(text='''Car Velocity :''')

        self.Button1 = tk.Button(self.Frame2)
        self.Button1.place(relx=0.288, rely=0.644, height=93, width=266)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#008080")
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(font=font15)
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''Start''')
        self.Button1.configure(command=lambda:clicked(self.Entry1.get()))

if __name__ == '__main__':
    vp_start_gui()





