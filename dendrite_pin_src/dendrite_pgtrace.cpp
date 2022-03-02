#include "pin.H"
#include <iostream>
#include <fstream>
#include <vector>
#include <dlfcn.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/socket.h>
#include "tool_macros.h"
using std::cerr;
using std::cout;
using std::endl;

struct FunctionTraceInfo {
    char function_name[64];
    int line_data;
    int level;
};

/* ===================================================================== */

INT32 Usage()
{
    cerr << "This pin tool inserts a user-written version of malloc() and free() into the application.\n"
            "\n";
    cerr << KNOB_BASE::StringKnobSummary();
    cerr << endl;
    return -1;
}

/* ===================================================================== */
/* Definitions for Probe mode */
/* ===================================================================== */

typedef uintptr_t Datum;
typedef void (*ErrFinishType)(const char *filename, int lineno, const char *funcname);
typedef void (*RecordEventType)(char *fname, int line, int level);
typedef void (*ExitType)(int code);
typedef void (*ParallelWorkerShutdownType)(int code, Datum arg);
typedef void (*TestHackType)(int code);

typedef typeof(malloc)* MallocType;
typedef typeof(free) *FreeType;
typedef typeof(read) *ReadType;
typedef typeof(write) *WriteType;
typedef typeof(pwrite) *PWriteType;
typedef typeof(recv) *RecvType;
typedef typeof(send) *SendType;


typedef typeof(dlopen)* DlopenType;
typedef typeof(dlsym)* DlsymType;
typedef typeof(dlerror)* DlerrorType;

std::vector<FunctionTraceInfo> infos;

RecordEventType recordEventTrampoline = 0;

ParallelWorkerShutdownType parallelWorkerShutdownWrapper = 0;
ParallelWorkerShutdownType origParallelWorkerShutdown = 0;

TestHackType testHackWrapper = 0;
TestHackType origTestHack = 0;

MallocType mallocWrapper = 0;
MallocType origMalloc = 0;

FreeType freeWrapper = 0;
FreeType origFree = 0;

ReadType readWrapper = 0;
ReadType origRead = 0;

WriteType writeWrapper = 0;
WriteType origWrite = 0;

PWriteType pwriteWrapper = 0;
PWriteType origPWrite = 0;

RecvType recvWrapper = 0;
RecvType origRecv = 0;

SendType sendWrapper = 0;
SendType origSend = 0;

ErrFinishType errFinishWrapper = 0;
ErrFinishType origErrFinish = 0;

ExitType exitWrapper = 0;
ExitType origExit = 0;

void* SharedObjectHandle = 0;

DlopenType AppDlopen = 0;
DlsymType AppDlsym   = 0;
DlerrorType AppDlerror   = 0;

typedef VOID (*SET_ORIG_FPTR)(
    ErrFinishType errFinishPtr,
    ExitType exitPtr,
    MallocType mallocPtr,
    FreeType freePtr,
    ReadType readPtr,
    WriteType writePtr,
    PWriteType pwritePtr,
    RecvType recvPtr,
    SendType sendPtr,
    ParallelWorkerShutdownType parallelWorkerShutdownPtr );

/* ===================================================================== */
/* Probe mode tool */
/* ===================================================================== */
void ErrFinishWrapperInTool( const char *filename, int lineno, const char *funcname ) {
    if( errFinishWrapper ) {
        (*errFinishWrapper)(filename, lineno, funcname);
        return;
    }
    (*origErrFinish)(filename, lineno, funcname);
}


VOID TestHackInTool( int val ) {
    (*testHackWrapper)( val );
}


void parallelWorkerShutdownInTool(int code, Datum arg) {
    (*parallelWorkerShutdownWrapper)( code, arg );
}

VOID *MallocWrapperInTool( size_t sz ) {
    if( mallocWrapper ) {
        return (*mallocWrapper)(sz);
    }
    return (*origMalloc)(sz);
}

VOID FreeWrapperInTool( void *ptr ) {
    if( freeWrapper ) {
        (*freeWrapper)(ptr);
        return;
    }
    (*origFree)(ptr);
}

ssize_t ReadWrapperInTool( int fd, void *buf, size_t count ) {
    if( readWrapper ) {
        return (*readWrapper)( fd, buf, count );
    }
    return (*origRead)( fd, buf, count );
}

ssize_t WriteWrapperInTool( int fd, const void *buf, size_t count ) {
    if( writeWrapper ) {
        return (*writeWrapper)( fd, buf, count );
    }
    return (*origWrite)( fd, buf, count );
}

ssize_t PWriteWrapperInTool( int fd, const void *buf, size_t count, off_t offset ) {
    if( pwriteWrapper ) {
        return (*pwriteWrapper)( fd, buf, count, offset );
    }
    return (*origPWrite)( fd, buf, count, offset );
}

ssize_t RecvWrapperInTool( int sockfd, void *buf, size_t len, int flags ) {
    if( recvWrapper ) {
        return (*recvWrapper)( sockfd, buf, len, flags );
    }
    return (*origRecv)( sockfd, buf, len, flags );
}

ssize_t SendWrapperInTool( int sockfd, const void *buf, size_t len, int flags ) {
    if( sendWrapper ) {
        return (*sendWrapper)( sockfd, buf, len, flags );
    }
    return (*origSend)( sockfd, buf, len, flags );
}

VOID ExitWrapperInTool(int code) {
    if( exitWrapper ) {
        (*exitWrapper)(code);
        return;
    }
    (*origExit)(code);
}

VOID TraceRtnCallback( ADDRINT *name_ptr, int line_data, int level )
{
    char *name = (char *) name_ptr;
    (*recordEventTrampoline)( name, line_data, level );
}

/* I'm calling dlopen before main.
 * Some malloc-free may be lost, of course.
 * But the earliest point you can call dlopen is after init of libc
 */
VOID MainRtnCallback()
{
    cout << "In main callback" << endl;
    // inject libmallocwrappers.so into application by executing application dlopen

    if( !AppDlopen ) {
        AppDlopen = dlopen;
    }
    if( !AppDlsym ) {
        AppDlsym = dlsym;
    }
    if( !AppDlerror ) {
        AppDlerror = dlerror;
    }

    char buff[256];
    cout << getcwd( buff, 256 ) << endl;
    SharedObjectHandle = AppDlopen("/hdd1/pin-3.21-98484-ge7cd811fd-gcc-linux/source/tools/Probes/obj-intel64/libpgwrapper.so", RTLD_LAZY);
    cout << "Before check." << endl;
    if( !SharedObjectHandle ) {
        cout << AppDlerror() << std::endl;
    }
    cout << "Before assert." << endl;
    ASSERTX(SharedObjectHandle);

    std::cout << "Dlloaded." << std::endl;
    // Get function pointers for the wrappers
    errFinishWrapper = ErrFinishType(AppDlsym(SharedObjectHandle, "errFinishWrapper"));
    ASSERTX(errFinishWrapper);
    exitWrapper = ExitType(AppDlsym(SharedObjectHandle, "exitWrapper"));
    ASSERTX(exitWrapper);
    recordEventTrampoline = RecordEventType(AppDlsym(SharedObjectHandle, "record_event_trampoline"));
    ASSERTX(recordEventTrampoline);

    testHackWrapper = TestHackType(AppDlsym(SharedObjectHandle, "testHackWrapper") );

    parallelWorkerShutdownWrapper = ParallelWorkerShutdownType(AppDlsym(SharedObjectHandle, "parallelWorkerShutdownWrapper") );
    ASSERTX( parallelWorkerShutdownWrapper );

    mallocWrapper = MallocType(AppDlsym(SharedObjectHandle, "mallocWrapper"));
    freeWrapper = FreeType(AppDlsym(SharedObjectHandle, "freeWrapper"));
    readWrapper = ReadType(AppDlsym(SharedObjectHandle, "readWrapper"));
    writeWrapper = WriteType(AppDlsym(SharedObjectHandle, "writeWrapper"));
    pwriteWrapper = PWriteType(AppDlsym(SharedObjectHandle, "pwriteWrapper"));
    recvWrapper = RecvType(AppDlsym(SharedObjectHandle, "recvWrapper"));
    sendWrapper = SendType(AppDlsym(SharedObjectHandle, "sendWrapper"));
    
    ASSERTX(mallocWrapper);
    ASSERTX(freeWrapper);
    ASSERTX(readWrapper);
    ASSERTX(writeWrapper);
    ASSERTX(pwriteWrapper);
    ASSERTX(recvWrapper);
    ASSERTX(sendWrapper);

    std::cout << "Set wrapper." << std::endl;

    // Send original function pointers to libmallocwrappers.so
    SET_ORIG_FPTR setOriginalFptr = (SET_ORIG_FPTR)AppDlsym(SharedObjectHandle, "SetOriginalFptr");
    (*setOriginalFptr)(origErrFinish, origExit, origMalloc, origFree, origRead, origWrite, origPWrite, origRecv, origSend, origParallelWorkerShutdown);

    cout << "Stuff set up." << endl;
}

VOID ImageLoad(IMG img, VOID* v)
{
    if (strstr(IMG_Name(img).c_str(), "libdl.so")) {
        // Get the function pointer for the application dlopen:
        // dlopen@@GLIBC_2.1 is the official, versioned name.
        //
        // The exact suffix must match the ABI of the libdl header files
        // this source code gets compiled against. Makefile/configure
        // trickery would be needed to figure this suffix out, so it
        // is simply hard-coded here.
        //
        // To keep the resulting binaries compatible with future libdl.so
        // versions, this code also checks for backwards compatibility
        // versions of the calls as they would be provided in such a
        // future version.

#if defined(TARGET_IA32E)
#define DLOPEN_VERSION "GLIBC_2.2.5"
#define DLSYM_VERSION "GLIBC_2.2.5"
#elif defined(TARGET_IA32)
#define DLOPEN_VERSION "GLIBC_2.1"
#define DLSYM_VERSION "GLIBC_2.0"
#else
#error symbol versions unknown for this target
#endif

        RTN dlopenRtn = RTN_FindByName(img, "dlopen@@" DLOPEN_VERSION);
        if (!RTN_Valid(dlopenRtn))
        {
            dlopenRtn = RTN_FindByName(img, "dlopen@" DLOPEN_VERSION);
        }

        if (!RTN_Valid(dlopenRtn))
        {
            // fallback for the cases in which symbols do not have a version
            dlopenRtn = RTN_FindByName(img, "dlopen");
        }

        ASSERTX(RTN_Valid(dlopenRtn));
        AppDlopen = DlopenType(RTN_Funptr(dlopenRtn));
        ASSERTX(AppDlopen);
        cout << "DlOpen set." << endl;

        // Get the function pointer for the application dlsym
        RTN dlsymRtn = RTN_FindByName(img, "dlsym@@" DLSYM_VERSION);
        if (!RTN_Valid(dlsymRtn))
        {
            dlsymRtn = RTN_FindByName(img, "dlsym@" DLSYM_VERSION);
        }
        if (!RTN_Valid(dlsymRtn))
        {
            // fallback for the cases in which symbols do not have a version
            dlsymRtn = RTN_FindByName(img, "dlsym");
        }

        ASSERTX(RTN_Valid(dlsymRtn));
        AppDlsym = DlsymType(RTN_Funptr(dlsymRtn));

        ASSERTX(AppDlsym);
        cout << "DlSym set." << endl;

        RTN dlerrorRtn = RTN_FindByName(img, "dlerror@@" DLSYM_VERSION);
        if (!RTN_Valid(dlerrorRtn))
        {
            dlerrorRtn = RTN_FindByName(img, "dlerror@" DLSYM_VERSION);
        }
        if (!RTN_Valid(dlerrorRtn))
        {
            // fallback for the cases in which symbols do not have a version
            dlerrorRtn = RTN_FindByName(img, "dlerror");
        }
        ASSERTX(RTN_Valid(dlerrorRtn));
        AppDlerror = DlerrorType(RTN_Funptr(dlerrorRtn));

        ASSERTX(AppDlerror);
        cout << "Dlerror set." << endl;

    }

    //if( strstr(IMG_Name(img).c_str(), "libc.so" ) ) {
        //cout << "Found libc." << endl;

        RTN mallocRtn = RTN_FindByName(img, "malloc");
        if( RTN_Valid( mallocRtn ) ) {
            if( !RTN_IsSafeForProbedReplacement( mallocRtn ) ) {
                cout << "Cannot replace malloc." << endl;
                exit(1);
            }
            origMalloc = (MallocType)RTN_ReplaceProbed(mallocRtn, AFUNPTR(MallocWrapperInTool));
        }

        RTN freeRtn = RTN_FindByName(img, "free");
        if( RTN_Valid( freeRtn ) ) {
            if( !RTN_IsSafeForProbedReplacement( freeRtn ) ) {
                cout << "Cannot replace free." << endl;
                exit(1);
            }

            origFree = (FreeType)RTN_ReplaceProbed(freeRtn, AFUNPTR(FreeWrapperInTool));
        }

        RTN readRtn = RTN_FindByName(img, "read");
        if( RTN_Valid( readRtn ) ) {
            if( !RTN_IsSafeForProbedReplacement( readRtn ) ) {
                cout << "Cannot replace read." << endl;
                exit(1);
            }

            origRead = (ReadType)RTN_ReplaceProbed(readRtn, AFUNPTR(ReadWrapperInTool));
        }

        RTN writeRtn = RTN_FindByName(img, "write");
        if( RTN_Valid( writeRtn ) ) {
            if( !RTN_IsSafeForProbedReplacement( writeRtn ) ) {
                cout << "Cannot replace write." << endl;
                exit(1);
            }

            origWrite = (WriteType)RTN_ReplaceProbed(writeRtn, AFUNPTR(WriteWrapperInTool));
        }

        RTN pwriteRtn = RTN_FindByName(img, "pwrite");
        if( RTN_Valid( pwriteRtn ) ) {
            if( !RTN_IsSafeForProbedReplacement( pwriteRtn ) ) {
                cout << "Cannot replace pwrite." << endl;
                exit(1);
            }

            origPWrite = (PWriteType)RTN_ReplaceProbed(pwriteRtn, AFUNPTR(PWriteWrapperInTool));
            cout << "Replaced pwrite in img: " << IMG_Name( img ) << endl;
        }

        RTN recvRtn = RTN_FindByName(img, "recv");
        if( RTN_Valid( recvRtn ) ) {
            if( !RTN_IsSafeForProbedReplacement( recvRtn ) ) {
                cout << "Cannot replace recv." << endl;
                exit(1);
            }

            origRecv = (RecvType)RTN_ReplaceProbed(recvRtn, AFUNPTR(RecvWrapperInTool));
        }

        RTN sendRtn = RTN_FindByName(img, "send");
        if( RTN_Valid( sendRtn ) ) {
            if( !RTN_IsSafeForProbedReplacement( sendRtn ) ) {
                cout << "Cannot replace send." << endl;
                exit(1);
            }

            origSend = (SendType)RTN_ReplaceProbed(sendRtn, AFUNPTR(SendWrapperInTool));
        }

        RTN exitRtn = RTN_FindByName(img, "exit");
        if( RTN_Valid(exitRtn) ) {
            if( !RTN_IsSafeForProbedReplacement(exitRtn) ) {
                cout << "Cannot replace proc_exit routine." << endl;
                exit(1);
            }
            origExit = (ExitType)RTN_ReplaceProbed(exitRtn, AFUNPTR(ExitWrapperInTool));
        }

    //}

    if( IMG_IsMainExecutable(img) ) {

        RTN errFinishRtn = RTN_FindByName(img, "errfinish");
        if( RTN_Valid(errFinishRtn) ) {
            if( !RTN_IsSafeForProbedReplacement(errFinishRtn) ) {
                cout << "Cannot replace errfinish routine." << endl;
                exit(1);
            }
            origErrFinish = (ErrFinishType)RTN_ReplaceProbed(errFinishRtn, AFUNPTR(ErrFinishWrapperInTool));
        }

        RTN parallelWorkerShutdownRtn = RTN_FindByName(img, "ParallelWorkerShutdown");
        if( !RTN_Valid( parallelWorkerShutdownRtn ) ) {
            cout << "Could not find parallel worker shutdown." << endl;
            exit(1);
        }
        ASSERTX( RTN_IsSafeForProbedReplacement( parallelWorkerShutdownRtn ) );
        origParallelWorkerShutdown = (ParallelWorkerShutdownType) RTN_ReplaceProbed( parallelWorkerShutdownRtn, AFUNPTR(parallelWorkerShutdownInTool) );

        RTN testHackRtn = RTN_FindByName(img, "beforeCloseDumpHack");
        if( RTN_Valid(testHackRtn) ) {
            if( !RTN_IsSafeForProbedReplacement(testHackRtn) ) {
                cout << "Could not set dump hack routine" << endl;
                exit(1);
            }
                origTestHack = (TestHackType)RTN_ReplaceProbed(testHackRtn, AFUNPTR(TestHackInTool));
            }


            // Need to override main. Can't not find it.
            RTN mainRtn = RTN_FindByName(img, "_main");
            if (!RTN_Valid(mainRtn)) mainRtn = RTN_FindByName(img, "main");

            if (!RTN_Valid(mainRtn))
            {
                cout << "Can't find the main routine in " << IMG_Name(img) << endl;
                exit(1);
            }

            RTN_InsertCallProbed(mainRtn, IPOINT_BEFORE, AFUNPTR(MainRtnCallback), IARG_END);

            //Override other custom functions
            for( FunctionTraceInfo &info : infos ) {
                IPOINT insertion_point = IPOINT_BEFORE;
                if( info.function_name[0] == '-' ) {
                    size_t len = strlen( info.function_name );
                    insertion_point = IPOINT_AFTER;
                    for( size_t i = 0; i < len; i++ ) {
                        // Go past for null char
                        info.function_name[i] = info.function_name[i+1];
                    }
                }

                if( insertion_point == IPOINT_BEFORE ) {
                    cout << "Tracing " << info.function_name << " at start of function." << endl;
                } else {
                    cout << "Tracing " << info.function_name << " at end of function." << endl;
                }
            
                RTN funcRtn = RTN_FindByName(img, info.function_name);
                if( !RTN_Valid(funcRtn) ) {
                    cout << "Cannot find " << info.function_name << endl;
                    exit(1);
                }
                if( !RTN_IsSafeForProbedReplacement(funcRtn) ) {
                    cout << info.function_name << " is not able to be replaced." << endl;
                    exit(1);
                }


                if( insertion_point == IPOINT_BEFORE ) {

                    RTN_InsertCallProbed(funcRtn, insertion_point, AFUNPTR(TraceRtnCallback), IARG_ADDRINT, info.function_name, IARG_UINT32, info.line_data, IARG_UINT32, info.level, IARG_END);
                    continue;
                } 
                // It's unclear to me how important it is that this function is correctly prototyped.
                PROTO proto = PROTO_Allocate(PIN_PARG(void), CALLINGSTD_DEFAULT, "proto", PIN_PARG(void *), PIN_PARG(void *), PIN_PARG_END() );

                RTN_InsertCallProbed(funcRtn, IPOINT_AFTER, AFUNPTR(TraceRtnCallback), IARG_PROTOTYPE, proto, IARG_ADDRINT, info.function_name, IARG_UINT32, info.line_data, IARG_UINT32, info.level, IARG_END);
                PROTO_Free( proto );
                
            }
            cout << "Override complete." << endl;
        }
    };

    void read_function_trace_info() {
        std::string fname = "functions_to_trace.cfg";
        std::ifstream cfg_file( fname.c_str() );
        while( true ) {
            FunctionTraceInfo info;
            cfg_file >> info.function_name;
            if( cfg_file.fail() ) {
                break;
            }
            cfg_file >> info.line_data;
            cfg_file >> info.level;
            infos.push_back( info );
        }
        cfg_file.close();

        cout << "Going to override " << infos.size() << " functions." << endl;
    }

    /* ===================================================================== */
    /* main */
    /* ===================================================================== */

    int main(int argc, CHAR* argv[])
    {

        read_function_trace_info();
        PIN_InitSymbols();

        if (PIN_Init(argc, argv))
        {
            return Usage();
        }

        IMG_AddInstrumentFunction(ImageLoad, 0);

        PIN_StartProgramProbed();

        return 0;
    }

    /* ===================================================================== */
    /* eof */
/* ===================================================================== */
