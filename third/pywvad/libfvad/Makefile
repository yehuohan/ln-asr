
WR_SRCS = \
	$(wildcard signal_processing/*.c) \
	$(wildcard vad/*.c)

WR_INCS = \
	-I./ \
	-Isignal_processing \
	-Ivad

.PHONY: pywvad

pywvad:
	gcc -shared -o pywvad.so $(WR_SRCS) $(WR_INCS)
