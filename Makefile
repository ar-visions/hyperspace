PROJECT := hyperspace

# use silver import through make
ifeq ($(strip $(IMPORT)),)
ifeq ($(wildcard ../silver),)
$(info required: tapestry build agent (invoked by Makefile) -- cloning to ../silver)
$(shell git clone https://github.com/ar-visions/silver ../silver)
$(error fetched silver-import -- please re-run make)
else
IMPORT := ../silver
endif
endif

include $(IMPORT)/bootstrap.mk
