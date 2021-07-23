SOURCES = \
    makedoc \
    make.jl \
    src/index.md \
    src/install.md \
    src/interpolation.md \
    src/library.md \
    src/notes.md

default: build/index.html

clean:
	rm -f *~ */*~ */*/*~

build/index.html: $(SOURCES)
	./makedoc

.PHOONY: default clean
