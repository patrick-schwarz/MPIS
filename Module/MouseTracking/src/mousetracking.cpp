#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <windows.h>
#include <math.h>
#include <mutex>

using namespace std;
namespace py = pybind11;
using namespace py::literals;

struct Measurement {
    int64_t x;
    int64_t y;
};

#define WM_INPUT 0x00FF
#define WM_HOOK 0x801
#define GWL_HINSTANCE       (-6)

std::mutex running_mutex;
bool running = false;

std::mutex measurement_mutex;
Measurement measurement ;

char rawinputdevices[256]; // string with number of raw input devices
float mouseDPI = 12000;

HWND  hwndMain;
HHOOK hookHandle;

vector<HANDLE > mouseInputs;

Measurement getMeasurement()
{
    Measurement result;

    measurement_mutex.lock();
    result = measurement;
    measurement_mutex.unlock();

    return result;
}
void setMeasurement(Measurement value)
{
    measurement_mutex.lock();
    measurement = value;
    measurement_mutex.unlock();
}

void resetToZero() {
    measurement_mutex.lock();

    measurement.x = 0;
    measurement.y = 0;

    measurement_mutex.unlock();
}


void setRunning(bool value){
    running_mutex.lock();
    running = value;
    running_mutex.unlock();
}

bool getRunning(){
    bool result = false;

    running_mutex.lock();
    result = running;
    running_mutex.unlock();

    return result;
}


RAWINPUT *rawMousePosition = NULL;
void UpdateMouseInput(LPARAM lParam) {
    if (rawMousePosition != NULL)
        free(rawMousePosition);
    LPBYTE lpb;
    UINT dwSize;
    GetRawInputData((HRAWINPUT) lParam, RID_INPUT, NULL, &dwSize, sizeof(RAWINPUTHEADER));

    lpb = (LPBYTE) malloc(sizeof(LPBYTE) * dwSize);
    if (lpb == NULL) {
        rawMousePosition = NULL;
    }

    GetRawInputData((HRAWINPUT) lParam, RID_INPUT, lpb, &dwSize, sizeof(RAWINPUTHEADER));
    //if (GetRawInputData((HRAWINPUT) lParam, RID_INPUT, lpb, &dwSize, sizeof(RAWINPUTHEADER)) != dwSize)
    //    py::print(py::str("GetRawInputData doesn't return correct size !\n"));

    rawMousePosition = (RAWINPUT *) lpb;
}


//	so we can access rawinput
void GetAllDevices() {
    RAWINPUTDEVICE Rid[50]; // allocate storage for 50 device (not going to need this many :) )

    UINT nDevices;
    PRAWINPUTDEVICELIST pRawInputDeviceList;
    if (GetRawInputDeviceList(NULL, &nDevices, sizeof(RAWINPUTDEVICELIST)) != 0) {
        return;
    }
    pRawInputDeviceList = (PRAWINPUTDEVICELIST) malloc(sizeof(RAWINPUTDEVICELIST) * nDevices);
    GetRawInputDeviceList(pRawInputDeviceList, &nDevices, sizeof(RAWINPUTDEVICELIST));

    UINT size = 256;
    TCHAR tBuffer[256] = {0};
    tBuffer[0] = '\0';
    RID_DEVICE_INFO rdi;
    rdi.cbSize = sizeof(RID_DEVICE_INFO);
    // do the job...
    for(int index = 0;index < nDevices;index++)
    {
        if(pRawInputDeviceList[index].dwType == RIM_TYPEMOUSE){
            if(GetRawInputDeviceInfo(pRawInputDeviceList[index].hDevice, RIDI_DEVICENAME, tBuffer, &size) > 0)
            {
                mouseInputs.push_back(pRawInputDeviceList[index].hDevice);
            }
        }
    }

    // after the job, free the RAWINPUTDEVICELIST
    free(pRawInputDeviceList);
    return;
}


LRESULT CALLBACK MainWndProc(HWND hwnd, UINT nMsg, WPARAM wParam, LPARAM lParam) {
    PAINTSTRUCT ps;              /* Also used during window drawing */
    RECT rc;              /* A rectangle used during drawing */
    HDC hdc;

    switch (nMsg) {
        case WM_CREATE: {
            GetAllDevices();
            RAWINPUTDEVICE Rid[1];
            Rid[0].usUsagePage = 0x01;
            Rid[0].usUsage = 0x02;
            Rid[0].dwFlags = 0;
            Rid[0].hwndTarget = hwnd;

            if (RegisterRawInputDevices(Rid, 1, sizeof(Rid[0])) == FALSE) {
                py::print(py::str("Registration of raw input devices failed"));
            }
            ///Starting Point
            Measurement m;

            m.x = 0;
            m.y = 0;

            setMeasurement(m);

            break;
        }
        case WM_DESTROY:
            if (rawMousePosition != NULL)
                free(rawMousePosition);
            PostQuitMessage(0);

            setRunning(false);

            break;
        case WM_INPUT: {

            int ItemIndex = 0;

            UpdateMouseInput(lParam);
            if (rawMousePosition != NULL &&
                rawMousePosition->header.dwType == RIM_TYPEMOUSE &&
                rawMousePosition->header.hDevice == mouseInputs[ItemIndex]    ) {
                HANDLE deviceHandle = rawMousePosition->header.hDevice;

                float move_x =   (float) rawMousePosition->data.mouse.lLastX ;
                float move_y =  (float) rawMousePosition->data.mouse.lLastY ;

                Measurement m = getMeasurement();
                m.x = (float) m.x + move_x;
                m.y = (float) m.y - move_y;
                setMeasurement(m);
            }
            break;
        }
        case WM_HOOK:
        {
            int ItemIndex = 0;
            if (rawMousePosition != NULL &&
                rawMousePosition->header.dwType == RIM_TYPEMOUSE &&
                rawMousePosition->header.hDevice == mouseInputs[ItemIndex] &&
                mouseInputs.size() > 2) {
                return 1;
            }
        }
        default:
            return DefWindowProc(hwnd, nMsg, wParam, lParam);
    }

    return 0;
}

static LRESULT CALLBACK MouseProc (int code, WPARAM wParam, LPARAM lParam)
{
    if (code >= 0 && SendMessage (hwndMain, WM_HOOK, wParam, lParam))
        return 1;

    return CallNextHookEx (hookHandle, code, wParam, lParam);
}

int run() {
    py::print(py::str("Starting mousetracking"));
    HWND hwndConsole = GetConsoleWindow();
    HINSTANCE hInstance = (HINSTANCE)GetWindowLong(hwndConsole, GWL_HINSTANCE);

    WNDCLASS wc      = {0};

	wc.lpfnWndProc   = MainWndProc;
	wc.hInstance     = hInstance;
	wc.lpszClassName = "WinTestWin";

    RegisterClass(&wc);
    hwndMain = CreateWindow(wc.lpszClassName,NULL,0,0,0,0,0,HWND_MESSAGE,NULL,hInstance,NULL);

    setRunning(true);

    py::gil_scoped_release release;

    hookHandle = SetWindowsHookEx (WH_MOUSE_LL, (HOOKPROC)MouseProc, 0, 0);

    MSG msg;
	while(GetMessage(&msg,hwndMain,0,0) !=0 && getRunning()){
        TranslateMessage(&msg);
        DispatchMessage(&msg);
	}

    UnhookWindowsHookEx (hookHandle);


    return msg.wParam;
}

void stop(){
    setRunning(false);
}
int64_t getX() {
    int64_t result = 0;

    measurement_mutex.lock();
    result = measurement.x;
    measurement_mutex.unlock();

    return result;
}
int64_t getY() {
    int64_t result = 0;

    measurement_mutex.lock();
    result = measurement.y;
    measurement_mutex.unlock();

    return result;
}

namespace py = pybind11;

PYBIND11_MODULE(mousetracking, m) {
  m.doc() = R"pbdoc(
        mousetracking module
        -----------------------
        .. currentmodule:: mousetracking
        .. autosummary::
           :toctree: _generate
           getX
    )pbdoc";

  m.def("getX", &getX, R"pbdoc(
        Returns x of raw mousinput
    )pbdoc");

  m.def("getY", &getY, R"pbdoc(
        Returns y of raw mousinput
    )pbdoc");


  m.def("run", &run, R"pbdoc(
        runs the mousetracking
    )pbdoc");

  m.def("resetToZero", &resetToZero, R"pbdoc(
        Resets the starting point to zero
    )pbdoc");

   m.def("getRunning", &getRunning, R"pbdoc(
       returns running status
   )pbdoc");

    m.def("stop", &stop, R"pbdoc(
        stops the mousetracking
    )pbdoc");

  m.attr("__version__") = "dev";
}