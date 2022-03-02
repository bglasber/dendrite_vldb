#include <imtrace/im_trace.hh>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/socket.h>

extern "C" {

    typedef uintptr_t Datum;
    typedef void (*ErrFinishType)(const char *filename, int lineno, const char *funcname);
    typedef void (*ParallelWorkerShutdownType)(int code, Datum arg);
    typedef void (*ExitType)(int code);
    typedef typeof(malloc) *MallocType;
    typedef typeof(free) *FreeType;
    typedef typeof(read) *ReadType;
    typedef typeof(write) *WriteType;
    typedef typeof(pwrite) *PWriteType;
    typedef typeof(recv) *RecvType;
    typedef typeof(send) *SendType;

    ErrFinishType errFinishFunc;
    ExitType exitFunc;
    MallocType mallocFunc;
    FreeType freeFunc;
    ReadType readFunc;
    WriteType writeFunc;
    PWriteType pwriteFunc;
    RecvType recvFunc;
    SendType sendFunc;

    ParallelWorkerShutdownType parallelWorkerShutdownFunc;

    void parallelWorkerShutdownWrapper(int code, Datum arg) {
        record_event( "ParallelWorkerShutdown", 0, 0 );
        dump_im_tracing();
        (*parallelWorkerShutdownFunc)(code,arg);
    }

    void errFinishWrapper(const char *filename, int lineno, const char *funcname ) {
        record_event( filename, lineno, 0 );
        (*errFinishFunc)(filename, lineno, funcname);
    }

    void exitWrapper(int code) {
        dump_im_tracing();
        (*exitFunc)(code);
    }

    void *mallocWrapper(size_t sz) {
        void *ret = (*mallocFunc)(sz);
        set_metric_info( MALLOC_METRIC_IDX, sz );
        return ret;
    }

    void freeWrapper(void *ptr) {
        (*freeFunc)(ptr);
        set_metric_info( FREE_METRIC_IDX, 1 );
    }

    ssize_t readWrapper( int fd, void *buf, size_t count ) {
        set_metric_info( READ_METRIC_IDX, count );
        return (*readFunc)( fd, buf, count );
    }

    ssize_t writeWrapper( int fd, const void *buf, size_t count ) {
        set_metric_info( WRITE_METRIC_IDX, count );
        return (*writeFunc)( fd, buf, count );
    }

    ssize_t pwriteWrapper( int fd, const void *buf, size_t count, off_t offset ) {
        set_metric_info( WRITE_METRIC_IDX, count );
        return (*pwriteFunc)( fd, buf, count, offset );
    }

    ssize_t recvWrapper( int sockfd, void *buf, size_t len, int flags ) {
        set_metric_info( RECV_METRIC_IDX, len );
        ssize_t ret = (*recvFunc)( sockfd, buf, len, flags );
        return ret;
    }
    ssize_t sendWrapper( int sockfd, const void *buf, size_t len, int flags ) {
        set_metric_info( SEND_METRIC_IDX, len );
        ssize_t ret = (*sendFunc)( sockfd, buf, len, flags );
        return ret;
    }

    // Avoids needing to deal with Pins linking crap
    void record_event_trampoline( char *filename, int line, int level) {
        record_event( filename, line, level );
    }

    void testHackWrapper() {
        printf( "WARNING!!!!: HACK: bumping epoch globally.\n" );
        set_global_dump();
    }

    void SetOriginalFptr(
        ErrFinishType errFinishF,
        ExitType exitF,
        MallocType mallocF,
        FreeType freeF,
        ReadType readF,
        WriteType writeF,
        PWriteType pwriteF,
        RecvType recvF,
        SendType sendF,
        ParallelWorkerShutdownType parallelWorkerShutdownF
     ) {
        errFinishFunc = errFinishF;
        exitFunc = exitF;
        mallocFunc = mallocF;
        freeFunc = freeF;
        readFunc = readF;
        writeFunc = writeF;
        pwriteFunc = pwriteF;
        recvFunc = recvF;
        sendFunc = sendF;
        parallelWorkerShutdownFunc = parallelWorkerShutdownF;
    }
}
