#!/bin/bash
set -e

mkdir -p docs/audit_reports

echo "Starting Subagent 1: Physics & Materials Science Audit..."
./scripts/smart_gemini.sh "You are a senior semiconductor physics expert. Review the PPT script at docs/ppt_presentation_script.md. Focus on:
1. Verify if the Conduction Band Offset (CBO) formula (CHI_PVK - CHI_ETL), Valence Band Offset (VBO) formula ((CHI_HTL + Eg_HTL) - CHI_PVK), and their ideal ranges (CBO [-0.1, 0.3] eV, VBO [1.7, 2.0] eV) are physically correct and accurately described.
2. Check if the defects and doping parameters, trap densities, built-in potential, and doping concentration limits (doping < 10^19 cm^-3) align with device physics.
3. Check for any physical inconsistencies in the PPT text or speaker notes.
Write your findings in Markdown to docs/audit_reports/physics_audit.md. When done, output a summary and finish." > docs/audit_reports/physics_audit.log 2>&1

echo "Starting Subagent 2: Codebase Consistency Audit..."
./scripts/smart_gemini.sh "You are a software architect. Review the PPT script at docs/ppt_presentation_script.md. Focus on:
1. Verify if the class names, method names, and file references (e.g. optimizer.py, knowledge.py, memory.py, data_loader.py, App.tsx, etc.) mentioned in the PPT match the actual implementation in the repository.
2. Verify if all file paths and hyperlinks in the markdown are correct.
3. Check if the description of the hybrid scoring formula matches the implementation in optimizer.py.
Write your findings in Markdown to docs/audit_reports/code_audit.md. When done, output a summary and finish." > docs/audit_reports/code_audit.log 2>&1

echo "Starting Subagent 3: Presentation Flow & Visuals Audit..."
./scripts/smart_gemini.sh "You are a presentation designer and visual artist. Review the PPT script at docs/ppt_presentation_script.md. Focus on:
1. Review the slide-by-slide narrative arc, checking if the transitions between background, architecture, core breakthroughs, benchmark results, operational mode, and summary are logical and engaging.
2. Audit the English Midjourney prompts. Check if they accurately reflect the selected 'Modern Academic Sci-Tech' style and effectively translate the slide concepts into visual metaphors.
3. Suggest visual design optimizations for the slides.
Write your findings in Markdown to docs/audit_reports/design_audit.md. When done, output a summary and finish." > docs/audit_reports/design_audit.log 2>&1

echo "All audit subagents have completed execution."
