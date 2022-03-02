CFLAGS = -std=c11 -O3 -Wall -fPIC -ggdb -fno-omit-frame-pointer
C++FLAGS = -std=c++17 -O3 -Wall -fPIC -ggdb -fno-omit-frame-pointer
CPPFLAGS = -DNDEBUG
JNIDIR = $(JAVA_HOME)

ifdef UNIT_TEST
CPPFLAGS += -DUNIT_TEST
endif

all: libimtrace.so libvotmodel.so

libimtrace.so : im_trace.o vot_model.o #org_neo4j_logging_shaded_log4j_core_DendriteTracer.o
	$(CXX) $(CPPFLAGS) $(C++FLAGS) -shared $^ -o libimtrace.so -L/hdd1/anaconda3/envs/dendrite/lib -lc -lpthread -lrt

libvotmodel.so : vot_model.o vot_model_interface.o
	$(CXX) $(CPPFLAGS) $(C++FLAGS) -shared $^ -o libvotmodel.so -L/hdd1/anaconda3/envs/dendrite/lib -lpthread

vot_model_interface.o : vot_model_interface.cc vot_model_interface.h
	$(CXX) -c $(CPPFLAGS) $(C++FLAGS) vot_model_interface.cc

%.o : %.cc %.hh
	$(CXX) -c $(CPPFLAGS) $(C++FLAGS) $< -o $@

#org_neo4j_logging_shaded_log4j_core_DendriteTracer.o : org_neo4j_logging_shaded_log4j_core_DendriteTracer.c org_neo4j_logging_shaded_log4j_core_DendriteTracer.h
#	$(CC) -c $(CPPFLAGS) $(CFLAGS) -I$(JNIDIR)/include -I$(JNIDIR)/include/linux $< -o $@

%.o : %.c %.h
	$(CC) -c $(CPPFLAGS) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o libimtrace.so libvotmodel.so

install :
	cp libimtrace.so /usr/local/lib
	cp libvotmodel.so /usr/local/lib
	mkdir -p /usr/local/include/imtrace
	cp im_trace.h /usr/local/include/imtrace
	cp vot_model_interface.h /usr/local/include/imtrace
	cp vot_model.hh /usr/local/include/imtrace
	cp trie.hh /usr/local/include/imtrace
