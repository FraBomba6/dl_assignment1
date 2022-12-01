DIRNAME := .
MDFOLDER = ./
MARKDOWNS := $(wildcard $(MDFOLDER)/*.md)
SOURCES := $(patsubst %.md,%.pdf,$(subst $(MDFOLDER),$(DIRNAME),$(MARKDOWNS)))

$(DIRNAME)/%.pdf: $(MDFOLDER)/%.md
	@echo "Building $@..."
	@pandoc 												   \
		--from markdown-implicit_figures \
		--template ./template.latex      \
		--listings 										   \
		-V babel=italian 							   \
		$^ 														   \
		-o $@

.PHONY : all

all: $(SOURCES)
