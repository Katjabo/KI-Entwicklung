# Legt die minimal erforderliche CMake-Version fest.
cmake_minimum_required(VERSION 3.10)

# Legt den Namen des Projekts fest.
project(ki_simulation VERSION 1.0)

# Setzt den C++ Standard für das Projekt. (C++17 ist ausreichend für die stabile Version)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Findet automatisch alle Quelldateien (.cpp) im 'src'-Verzeichnis.
file(GLOB SRC_FILES "src/*.cpp")

# Definiert die ausführbare Datei.
add_executable(${PROJECT_NAME} main.cpp ${SRC_FILES})

# Fügt das 'src'-Verzeichnis als Include-Pfad für das Target hinzu.
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

# --- NEUER ABSCHNITT: Automatisches Kopieren der Abhängigkeiten ---

# Diese Logik wird nur auf Windows mit dem GNU C++ Compiler (MinGW) ausgeführt.
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

    # Finde das Verzeichnis, in dem sich der g++ Compiler befindet (z.B. C:/msys64/ucrt64/bin)
    get_filename_component(MINGW_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)

    # Erstelle eine Liste der benötigten DLL-Dateien.
    set(REQUIRED_DLLS
        libstdc++-6.dll
        libgcc_s_seh-1.dll
        libwinpthread-1.dll
    )

    # Gehe durch jede DLL in der Liste...
    foreach(DLL_NAME ${REQUIRED_DLLS})
        
        # Erstelle den vollen Pfad zur Quell-DLL
        set(DLL_SOURCE_PATH "${MINGW_BIN_DIR}/${DLL_NAME}")

        # Füge einen benutzerdefinierten Befehl hinzu, der NACH dem Kompilieren
        # von "ki_simulation" ausgeführt wird.
        add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            # Der Befehl ist CMake's plattformunabhängiger Kopiervorgang
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            # Quelle
            "${DLL_SOURCE_PATH}"
            # Ziel: Das Verzeichnis, in dem die .exe-Datei liegt
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
            # Kommentar, der während des Build-Prozesses angezeigt wird
            COMMENT "Kopiere Abhängigkeit ${DLL_NAME}"
        )

    endforeach()

endif()