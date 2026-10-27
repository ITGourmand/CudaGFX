# Makefile: multi-platform build of the `cudagfx` tool (native Linux, and Windows cross-compiled with mingw-w64).
#
# Layout convention: every src/<name>.cpp becomes the executable <name> (here: src/cudagfx.cpp -> cudagfx);
# every include/**/*.cpp is compiled once and linked into all the executables (the transpiler library).
#
# Targets (run `make <target>`):
#   all / linux / windows   build the tool into .build/<os>/bin/
#   unit                    build and run the front-end unit tests (tests/unit, Linux only)
#   test                    unit tests, then the full suite tests/run_tests.sh (needs bash; see its header for options)
#   clean                   remove .build/
#
# The tool is built with -std=c++17 (no GNU extensions, constraint C1 of the specification) and -Werror:
# keep the code warning-free.

# --- top level (no TARGET_OS): dispatch to one sub-make per platform ----------------------------------------------
ifeq ($(TARGET_OS),)

all: linux windows

linux:
	@$(MAKE) TARGET_OS=linux

windows:
	@$(MAKE) TARGET_OS=windows

clean:
	@$(MAKE) TARGET_OS=linux clean
	@$(MAKE) TARGET_OS=windows clean

# Front-end unit tests, then the full suite (needs bash and g++; shaderc/glslang are optional)
test:
	@$(MAKE) TARGET_OS=linux unit
	@tests/run_tests.sh

.PHONY: all linux windows clean test

else

# --- per-platform build (TARGET_OS = linux | windows) ----------------------------------------------------------------

BUILD_DIR  := .build/$(TARGET_OS)
OBJ_DIR    := $(BUILD_DIR)/obj
BIN_DIR    := $(BUILD_DIR)/bin
SRC_DIR    := src
INC_DIR    := include
ASSETS_DIR := assets

ifeq ($(TARGET_OS), windows)
    CC := x86_64-w64-mingw32-g++
    WINDRES := x86_64-w64-mingw32-windres
    EXE_EXT := .exe
    CFLAGS := -std=c++17 -O3 -Wall -Wextra -Wpedantic -Werror -MMD -MP -I$(INC_DIR) -DWIN32 -D_WINDOWS
    LDFLAGS := -static -static-libgcc -static-libstdc++
else ifeq ($(TARGET_OS), linux)
    CC := g++
    WINDRES :=
    EXE_EXT :=
    CFLAGS := -std=c++17 -O3 -Wall -Wextra -Wpedantic -Werror -MMD -MP -I$(INC_DIR)
    LDFLAGS := -static-libgcc -static-libstdc++ -lpthread
else
    $(error Unknown TARGET_OS. Use "linux" or "windows")
endif

# Library sources (compiled once, shared by every executable) and application sources (one executable each)
INC_SRCS   := $(wildcard $(INC_DIR)/*.cpp) $(wildcard $(INC_DIR)/CudaGFX/*.cpp)
INC_OBJS   := $(patsubst $(INC_DIR)/%.cpp,$(OBJ_DIR)/inc_%.o,$(INC_SRCS))

APP_SRCS   := $(wildcard $(SRC_DIR)/*.cpp)
STEMS      := $(patsubst $(SRC_DIR)/%.cpp,%,$(APP_SRCS))

# Header dependencies (-MMD): editing a .hpp recompiles the translation units that include it.
DEPS       := $(INC_OBJS:.o=.d) $(patsubst %,$(OBJ_DIR)/%.d,$(STEMS))

# Front-end unit tests (Linux only: the executable runs on the build machine)
UNIT_SRCS  := $(wildcard tests/unit/*.cpp)

.DEFAULT_GOAL := all
all: $(STEMS:%=$(BIN_DIR)/%$(EXE_EXT))

ifeq ($(TARGET_OS), windows)

# Windows resource file: assets/<name>.rc if it exists, otherwise assets/app.rc (if any)
GET_RC = $(if $(wildcard $(ASSETS_DIR)/$(1).rc),$(ASSETS_DIR)/$(1).rc,$(wildcard $(ASSETS_DIR)/app.rc))

# BUILD_TARGET(<name>): link rule of one executable (with its resource object when an .rc file exists)
define BUILD_TARGET
ifneq ($(call GET_RC,$(1)),)
$(OBJ_DIR)/$(1)_resource.o: $(call GET_RC,$(1))
	@mkdir -p $$(@D)
	@echo "🎨 Compiling Resource [$(call GET_RC,$(1))] for $(1) -> $$(@)"
	$$(WINDRES) $$< -O coff -o $$(@)

$(BIN_DIR)/$(1)$(EXE_EXT): $(OBJ_DIR)/$(1).o $(INC_OBJS) $(OBJ_DIR)/$(1)_resource.o
	@mkdir -p $$(@D)
	@echo "🔗 Linking [$(TARGET_OS)] -> $$(@)"
	$$(CC) $$^ -o $$(@) $$(LDFLAGS)
else
$(BIN_DIR)/$(1)$(EXE_EXT): $(OBJ_DIR)/$(1).o $(INC_OBJS)
	@mkdir -p $$(@D)
	@echo "🔗 Linking [$(TARGET_OS)] -> $$(@)"
	$$(CC) $$^ -o $$(@) $$(LDFLAGS)
endif
endef

else

# BUILD_TARGET(<name>): link rule of one executable
define BUILD_TARGET
$(BIN_DIR)/$(1)$(EXE_EXT): $(OBJ_DIR)/$(1).o $(INC_OBJS)
	@mkdir -p $$(@D)
	@echo "🔗 Linking [$(TARGET_OS)] -> $$(@)"
	$$(CC) $$^ -o $$(@) $$(LDFLAGS)
endef

endif

# Compilation rules: application sources, then library sources (object names are prefixed with inc_)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo "🔨 Compiling App [$(TARGET_OS)] -> $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/inc_%.o: $(INC_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo "🔨 Compiling Lib [$(TARGET_OS)] -> $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(foreach stem,$(STEMS),$(eval $(call BUILD_TARGET,$(stem))))

$(BIN_DIR)/unit_tests$(EXE_EXT): $(UNIT_SRCS) $(INC_OBJS)
	@mkdir -p $(@D)
	@echo "🧪 Linking unit tests [$(TARGET_OS)] -> $@"
	$(CC) $(CFLAGS) -Itests/unit $^ -o $@ $(LDFLAGS)

unit: $(BIN_DIR)/unit_tests$(EXE_EXT)
	@$<

clean:
	rm -rf .build

-include $(DEPS)

.PHONY: all clean unit

endif
