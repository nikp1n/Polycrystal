"""Запуск окна с правами рабочего стола для drag-and-drop из Проводника."""
import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import sys


def relaunch_on_desktop(script):
    """Возвращает True, если вместо повышенного процесса запущена обычная копия.

    Windows запрещает OLE drag-and-drop из обычного Explorer в повышенное окно.
    Приложению права администратора не нужны: запускаем его с токеном оболочки.
    Используется документированный PROC_THREAD_ATTRIBUTE_PARENT_PROCESS:
    https://devblogs.microsoft.com/oldnewthing/20190425-00/?p=102443
    """
    if sys.platform != 'win32':
        return False
    shell = ctypes.WinDLL('shell32', use_last_error=True)
    if not shell.IsUserAnAdmin():
        return False
    if '--desktop-launch' in sys.argv:
        raise OSError('Не удалось запустить приложение с обычными правами. '
                      'Закройте PyCharm и запустите его без прав администратора.')
    _launch_from_shell(Path(script).resolve())
    return True


def _launch_from_shell(script):
    """Создать процесс с токеном Explorer, не меняя UAC и фильтры сообщений."""
    c, w = ctypes, wintypes
    kernel = c.WinDLL('kernel32', use_last_error=True)
    user = c.WinDLL('user32', use_last_error=True)

    class StartupInfo(c.Structure):
        _fields_ = [
            ('cb', w.DWORD), ('lpReserved', w.LPWSTR),
            ('lpDesktop', w.LPWSTR), ('lpTitle', w.LPWSTR),
            ('dwX', w.DWORD), ('dwY', w.DWORD),
            ('dwXSize', w.DWORD), ('dwYSize', w.DWORD),
            ('dwXCountChars', w.DWORD), ('dwYCountChars', w.DWORD),
            ('dwFillAttribute', w.DWORD), ('dwFlags', w.DWORD),
            ('wShowWindow', w.WORD), ('cbReserved2', w.WORD),
            ('lpReserved2', c.POINTER(c.c_ubyte)),
            ('hStdInput', w.HANDLE), ('hStdOutput', w.HANDLE),
            ('hStdError', w.HANDLE),
        ]

    class StartupInfoEx(c.Structure):
        _fields_ = [('StartupInfo', StartupInfo), ('lpAttributeList', c.c_void_p)]

    class ProcessInfo(c.Structure):
        _fields_ = [('hProcess', w.HANDLE), ('hThread', w.HANDLE),
                    ('dwProcessId', w.DWORD), ('dwThreadId', w.DWORD)]

    user.GetShellWindow.argtypes = []
    user.GetShellWindow.restype = w.HWND
    user.GetWindowThreadProcessId.argtypes = [w.HWND, c.POINTER(w.DWORD)]
    user.GetWindowThreadProcessId.restype = w.DWORD
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.CloseHandle.restype = w.BOOL
    kernel.InitializeProcThreadAttributeList.argtypes = [
        c.c_void_p, w.DWORD, w.DWORD, c.POINTER(c.c_size_t)]
    kernel.InitializeProcThreadAttributeList.restype = w.BOOL
    kernel.UpdateProcThreadAttribute.argtypes = [
        c.c_void_p, w.DWORD, c.c_size_t, c.c_void_p, c.c_size_t,
        c.c_void_p, c.c_void_p]
    kernel.UpdateProcThreadAttribute.restype = w.BOOL
    kernel.DeleteProcThreadAttributeList.argtypes = [c.c_void_p]
    kernel.DeleteProcThreadAttributeList.restype = None
    kernel.CreateProcessW.argtypes = [
        w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p, w.BOOL, w.DWORD,
        c.c_void_p, w.LPCWSTR, c.POINTER(StartupInfoEx), c.POINTER(ProcessInfo)]
    kernel.CreateProcessW.restype = w.BOOL

    pid = w.DWORD()
    desktop = user.GetShellWindow()
    if not desktop or not user.GetWindowThreadProcessId(desktop, c.byref(pid)):
        raise OSError('Не найден рабочий стол Windows. Запустите PyCharm без прав администратора.')
    parent = w.HANDLE(kernel.OpenProcess(0x0080, False, pid.value))  # PROCESS_CREATE_PROCESS
    if not parent:
        raise c.WinError(c.get_last_error())
    initialized = False
    try:
        size = c.c_size_t()
        kernel.InitializeProcThreadAttributeList(None, 1, 0, c.byref(size))
        attributes = c.create_string_buffer(size.value)
        if not kernel.InitializeProcThreadAttributeList(attributes, 1, 0, c.byref(size)):
            raise c.WinError(c.get_last_error())
        initialized = True
        if not kernel.UpdateProcThreadAttribute(
                attributes, 0, 0x00020000, c.byref(parent), c.sizeof(parent), None, None):
            raise c.WinError(c.get_last_error())
        startup = StartupInfoEx()
        startup.StartupInfo.cb = c.sizeof(startup)
        startup.lpAttributeList = c.cast(attributes, c.c_void_p)
        process = ProcessInfo()
        command = c.create_unicode_buffer(subprocess.list2cmdline([
            sys.executable, '-B', str(script), '--desktop-launch', *sys.argv[1:]]))
        # CREATE_NO_WINDOW скрывает только консоль Python; окно Qt остаётся видимым.
        if not kernel.CreateProcessW(
                sys.executable, command, None, None, False, 0x08080000,
                None, str(script.parent), c.byref(startup), c.byref(process)):
            raise c.WinError(c.get_last_error())
        kernel.CloseHandle(process.hThread)
        kernel.CloseHandle(process.hProcess)
    finally:
        if initialized:
            kernel.DeleteProcThreadAttributeList(attributes)
        kernel.CloseHandle(parent)
