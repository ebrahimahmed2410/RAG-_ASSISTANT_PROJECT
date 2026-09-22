# Dataset Documentation: Computer Science Educational Documents

## 1. Domain Overview
This dataset contains academic lecture handouts and reference documents for undergraduate Computer Science courses. The curriculum encompasses foundational topics across four core domains:

1. **CS101: Data Structures and Algorithms** (`cs101_data_structures.pdf`)
   - Topics: Arrays, Linked Lists, Binary Search Trees, Balanced Trees (AVL/Red-Black), Hash Tables, Graph Traversal (BFS/DFS), Big-O Asymptotic Complexity, Sorting Algorithms (Quicksort, Mergesort, Heapsort).
2. **CS102: Operating Systems Concepts** (`cs102_operating_systems.pdf`)
   - Topics: Processes, Threads, Process Scheduling (Round Robin, Priority, Multilevel Feedback Queues), Inter-Process Communication (IPC), Synchronization and Semaphores, Deadlock (Coffman Conditions, Banker's Algorithm), Virtual Memory, Paging, TLB, Page Replacement Algorithms.
3. **CS103: Database Management Systems** (`cs103_database_systems.pdf`)
   - Topics: Relational Model, Relational Algebra, SQL Fundamentals, Database Normalization (1NF, 2NF, 3NF, BCNF), Transaction Processing, ACID Properties, Serializability, Concurrency Control (Two-Phase Locking), Indexing with B-Trees and Hash Indexes.
4. **CS104: Computer Networks and Protocols** (`cs104_computer_networks.pdf`)
   - Topics: The OSI 7-Layer Reference Model vs. TCP/IP Architecture, Physical & Data Link Layers (Ethernet, CSMA/CD), Network Layer (IP Addressing, Subnetting, CIDR, Distance-Vector & Link-State Routing), Transport Layer (TCP 3-Way Handshake, Flow Control, Congestion Control vs. UDP), Application Layer Protocols (DNS, HTTP/1.1, HTTP/2, HTTPS/TLS).

## 2. Directory Structure
```text
data/
├── README.md               # This documentation file
├── raw/                    # Raw source PDF files extracted by the RAG ingestion pipeline
│   ├── cs101_data_structures.pdf
│   ├── cs102_operating_systems.pdf
│   ├── cs103_database_systems.pdf
│   └── cs104_computer_networks.pdf
└── processed/              # Processed chunks, extraction summaries, and metadata artifacts
```

## 3. Ingestion & Preprocessing Guidelines
When adding new documents to `data/raw/`:
1. Ensure the documents are text-based PDFs (or searchable OCR-processed PDFs).
2. The PyPDF loader will inspect and parse all pages, counting total characters, words, and pages.
3. Chunks are generated using a sliding window strategy (`chunk_size=800` characters, `chunk_overlap=150` characters) to maintain contextual continuity across chunk boundaries.
4. Metadata attached to each chunk:
   - `document`: Base filename of the PDF (e.g. `cs101_data_structures.pdf`).
   - `page`: 1-indexed page number in the original PDF.
   - `chunk_id`: Unique identifier combining document, page, and chunk index.

## 4. Academic Integrity & Sourcing
The provided materials represent standard, peer-reviewed computer science curriculum knowledge as taught in accredited university computer science departments. You may replace or supplement these files with your university's official syllabi, textbook excerpts, or course handouts.
