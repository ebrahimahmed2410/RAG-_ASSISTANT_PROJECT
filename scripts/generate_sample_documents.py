"""
Generate authentic, high-quality Computer Science educational PDF documents
for the RAG Document Assistant dataset.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def create_pdf(filename: str, title: str, pages_content: list):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=12,
    )

    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=8,
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#2D3748'),
        leftIndent=15,
        spaceAfter=4,
    )

    story = []

    for page_idx, (page_title, sections) in enumerate(pages_content):
        if page_idx > 0:
            story.append(PageBreak())

        # Header for the page
        story.append(Paragraph(f"{title} &mdash; {page_title}", title_style))
        story.append(Spacer(1, 8))

        for sec_heading, paragraphs in sections:
            story.append(Paragraph(sec_heading, h2_style))
            for p_text in paragraphs:
                if p_text.startswith("•") or p_text.startswith("-"):
                    story.append(Paragraph(p_text, bullet_style))
                else:
                    story.append(Paragraph(p_text, body_style))
            story.append(Spacer(1, 6))

    doc.build(story)
    print(f"[+] Generated: {filename} ({len(pages_content)} pages)")


def generate_all_documents():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(data_dir, exist_ok=True)

    # 1. CS101: Data Structures and Algorithms
    cs101_pages = [
        (
            "Page 1: Linear Data Structures and Asymptotic Foundations",
            [
                (
                    "1. Arrays and Dynamic Arrays",
                    [
                        "An array is a contiguous block of memory storing elements of homogenous type. In a static array, size is fixed at allocation time. Element access is achieved via pointer arithmetic: Address(A[i]) = BaseAddress + i * sizeof(Element), providing deterministic O(1) time complexity.",
                        "Dynamic arrays (such as Python lists or C++ std::vector) automatically resize when capacity is reached. The common growth factor is 2. Resizing requires allocating a new array of double the capacity and copying all n elements, an O(n) operation. However, amortized analysis over a sequence of n insertions shows that the amortized cost per insertion is O(1).",
                    ]
                ),
                (
                    "2. Linked Lists",
                    [
                        "A linked list is a linear collection of data elements where order is not given by physical placement in memory. Instead, each element points to the next. In a Singly Linked List, each node contains a value and a 'next' pointer. In a Doubly Linked List, each node contains 'next' and 'prev' pointers.",
                        "• Insertion at the head is O(1) time complexity.",
                        "• Insertion at arbitrary position requires traversal, taking O(n) time.",
                        "• Search requires linear traversal, resulting in O(n) worst-case time.",
                        "• Linked lists avoid contiguous memory requirements and prevent costly reallocation copying.",
                    ]
                ),
                (
                    "3. Stacks and Queues",
                    [
                        "A Stack operates on a Last-In, First-Out (LIFO) protocol. Key operations are push, pop, and peek, all executing in O(1) time. Stacks are used in runtime call stacks, expression evaluation (Shunting-yard algorithm), and backtrack algorithms.",
                        "A Queue operates on a First-In, First-Out (FIFO) discipline. Operations include enqueue and dequeue in O(1) time. Implementations frequently use circular buffers or doubly linked lists to avoid O(n) shifting. Applications include breadth-first search and task scheduling buffers.",
                    ]
                )
            ]
        ),
        (
            "Page 2: Non-Linear Structures: Trees and Balancing",
            [
                (
                    "1. Binary Search Trees (BST)",
                    [
                        "A Binary Search Tree is a hierarchical structure where each node has at most two children. The BST invariant mandates that for any node X, all values in the left subtree are strictly less than X, and all values in the right subtree are strictly greater than X.",
                        "In an optimally balanced BST, search, insertion, and deletion operate in O(log n) time. However, if keys are inserted in sorted order, the BST degenerates into a linked list with height O(n), causing worst-case operations to degrade to O(n).",
                        "Traversals: In-order traversal (Left, Root, Right) of a BST outputs elements in monotonically ascending order. Pre-order is used for serialization, and Post-order is used for deletion and tree destruction.",
                    ]
                ),
                (
                    "2. Self-Balancing Trees: AVL and Red-Black Trees",
                    [
                        "To prevent tree degeneration, self-balancing binary search trees maintain logarithmic height.",
                        "• AVL Trees: For every node, the height difference between left and right subtrees (the Balance Factor) must be in {-1, 0, +1}. When an insertion or deletion violates this, tree rotations (Left Rotation, Right Rotation, Left-Right Rotation, or Right-Left Rotation) restore balance in O(log n) time.",
                        "• Red-Black Trees: Nodes are colored Red or Black. The root is Black, no two consecutive Red nodes are allowed, and every path from root to leaf contains the identical number of Black nodes. Red-Black trees require fewer rotations during mutations than AVL trees, making them standard for system library maps (such as C++ std::map).",
                    ]
                )
            ]
        ),
        (
            "Page 3: Hash Tables and Graph Algorithms",
            [
                (
                    "1. Hash Tables and Collision Resolution",
                    [
                        "A Hash Table maps keys to indices in an array using a hash function: index = hash(key) mod capacity. An effective hash function distributes keys uniformly across slots to minimize collisions.",
                        "Collision Resolution Strategies:",
                        "• Separate Chaining: Each bucket holds a linked list or small balanced tree of entries that hash to the same bucket. Average lookup time is O(1 + alpha), where alpha = n / m is the load factor.",
                        "• Open Addressing: All elements reside directly in the table. On collision, probe sequences are examined: Linear Probing (hash(k) + i mod m), Quadratic Probing (hash(k) + c1*i + c2*i^2 mod m), or Double Hashing (hash1(k) + i * hash2(k) mod m).",
                    ]
                ),
                (
                    "2. Graph Representations and Traversal",
                    [
                        "A graph G = (V, E) comprises vertices V and edges E. Graphs are represented using:",
                        "• Adjacency Matrix: A 2D array of size |V| x |V|. Edge lookup is O(1), but memory complexity is O(|V|^2), inefficient for sparse graphs.",
                        "• Adjacency List: An array or dictionary of lists, storing adjacent neighbors for each vertex. Space complexity is O(|V| + |E|).",
                        "Traversals:",
                        "• Breadth-First Search (BFS): Explores nodes level by level using a FIFO queue. Time complexity is O(|V| + |E|). BFS identifies the shortest path in unweighted graphs.",
                        "• Depth-First Search (DFS): Explores branches deeply using recursion or a LIFO stack. Time complexity is O(|V| + |E|). Used for cycle detection, topological sorting of DAGs, and strongly connected components.",
                    ]
                )
            ]
        ),
        (
            "Page 4: Asymptotic Complexity and Sorting Algorithms",
            [
                (
                    "1. Asymptotic Notations",
                    [
                        "Computational complexity assesses algorithm efficiency as input size n approaches infinity.",
                        "• Big-O (O): Asymptotic upper bound. f(n) = O(g(n)) if there exist constants c > 0 and n0 such that f(n) <= c * g(n) for all n >= n0.",
                        "• Big-Omega (Ω): Asymptotic lower bound. f(n) = Ω(g(n)) if f(n) >= c * g(n) for all n >= n0.",
                        "• Big-Theta (Θ): Asymptotic tight bound. f(n) = Θ(g(n)) if f(n) is both O(g(n)) and Ω(g(n)).",
                    ]
                ),
                (
                    "2. Comparison Sorting Algorithms",
                    [
                        "• Quicksort: Divide-and-conquer algorithm. Selects a pivot, partitions array into elements less than and greater than pivot, and recursively sorts subarrays. Average time complexity is O(n log n). Worst-case time is O(n^2) when pivot is poorly chosen (e.g., smallest/largest in already sorted array). Mitigated by Randomized Pivot selection.",
                        "• Mergesort: Divides array into halves, recursively sorts, and merges sorted halves. Guaranteed O(n log n) worst-case time complexity. Stable sort, but requires O(n) auxiliary space.",
                        "• Heapsort: Builds a max-heap in O(n) time, then repeatedly extracts maximum element in O(log n) time. In-place sorting with O(n log n) worst-case time complexity, but not stable.",
                    ]
                ),
                (
                    "3. Binary Search",
                    [
                        "Binary Search locates a target element in a sorted array by repeatedly halving the search interval. At each step, the target is compared with the median element.",
                        "Recurrence relation: T(n) = T(n/2) + O(1). By the Master Theorem, time complexity is O(log n). Space complexity is O(1) iterative or O(log n) recursive.",
                    ]
                )
            ]
        )
    ]

    # 2. CS102: Operating Systems Concepts
    cs102_pages = [
        (
            "Page 1: Processes, Threads, and Concurrency",
            [
                (
                    "1. The Process Model and Process Control Block",
                    [
                        "A process is an active program in execution. The operating system maintains metadata for each process in a Process Control Block (PCB). The PCB contains: Process Identification (PID), Process State, Program Counter (PC), CPU Registers, CPU Scheduling Information (priority), Memory-Management Information (page tables), and I/O Status Information.",
                        "Process Lifecycle States:",
                        "• New: Process is being created.",
                        "• Ready: Process is loaded in memory and waiting for CPU allocation.",
                        "• Running: Instructions are being executed by the CPU core.",
                        "• Waiting/Blocked: Process is waiting for an I/O completion or event.",
                        "• Terminated: Process has completed execution.",
                        "Context Switch: The mechanism of saving the state of the active process in its PCB and restoring the state of another scheduled process. Context switches incur pure system overhead.",
                    ]
                ),
                (
                    "2. Threads and Multithreading",
                    [
                        "A thread is the smallest unit of CPU utilization within a process. Multiple threads of the same process share the code section, data section, and open files (OS resources). However, each thread retains its own private Thread ID, Program Counter, Register set, and Stack.",
                        "Benefits of multithreading: Responsiveness in interactive applications, resource sharing, economy (creating threads is faster than processes), and scalable utilization of multi-core processors.",
                    ]
                )
            ]
        ),
        (
            "Page 2: CPU Scheduling and Synchronization",
            [
                (
                    "1. CPU Scheduling Algorithms",
                    [
                        "• First-Come, First-Served (FCFS): Non-preemptive. Simple FIFO queue, but suffers from the Convoy Effect where short processes wait behind long CPU bursts.",
                        "• Shortest Job First (SJF) & Shortest Remaining Time First (SRTF): Schedules the process with minimal CPU burst duration. Provably optimal in minimizing average waiting time, but susceptible to starvation of long jobs.",
                        "• Round Robin (RR): Preemptive scheduling where each ready process receives a discrete time slice (time quantum q). If quantum is excessively large, RR degenerates to FCFS; if quantum is too small, excessive context-switch overhead degrades performance.",
                        "• Multilevel Feedback Queue (MLFQ): Multiple priority queues with adaptive demotion for CPU-intensive processes and promotion for I/O-bound processes.",
                    ]
                ),
                (
                    "2. The Critical Section Problem and Synchronization",
                    [
                        "A race condition occurs when multiple threads concurrently access and mutate shared data, and the outcome depends on execution timing. The Critical Section must satisfy three criteria: Mutual Exclusion, Progress, and Bounded Waiting.",
                        "Primitives:",
                        "• Mutex Locks: Binary flag providing mutual exclusion via acquire() and release().",
                        "• Semaphores: Integer variable S accessed through wait() (P) and signal() (V). A counting semaphore controls access to a finite set of resources.",
                    ]
                )
            ]
        ),
        (
            "Page 3: Deadlock Principles and Management",
            [
                (
                    "1. The Four Coffman Deadlock Conditions",
                    [
                        "A deadlock situation in an operating system occurs if and only if all four Coffman conditions hold simultaneously:",
                        "1. Mutual Exclusion: At least one resource must be held in a non-shareable mode.",
                        "2. Hold and Wait: A process must hold at least one resource while waiting to acquire additional resources held by other processes.",
                        "3. No Preemption: Resources cannot be forcibly taken from a process holding them; they must be released voluntarily.",
                        "4. Circular Wait: A closed chain of processes exists such that each process holds at least one resource needed by the next process in the cycle.",
                    ]
                ),
                (
                    "2. Deadlock Handling Strategies",
                    [
                        "• Deadlock Prevention: Eliminates at least one of the four Coffman conditions. For example, enforcing a total resource ordering eliminates Circular Wait.",
                        "• Deadlock Avoidance: The OS inspects resource allocation requests dynamically using the Banker's Algorithm. The system determines if granting a request leaves the system in a 'Safe State' (a sequence of process completions exists where all demands can be met).",
                        "• Deadlock Detection and Recovery: Employs a Wait-For Graph (WFG) to detect cycles. Recovery is accomplished via process termination or selective resource preemption with rollback.",
                    ]
                )
            ]
        ),
        (
            "Page 4: Virtual Memory, Paging, and Page Replacement",
            [
                (
                    "1. Paging Architecture and Address Translation",
                    [
                        "Virtual memory decouples logical addresses viewed by user programs from physical RAM addresses. Physical memory is segmented into fixed-size frames, and logical memory is divided into equal-sized pages (standard 4KB).",
                        "The Memory Management Unit (MMU) translates a logical address (Page Number p, Offset d) to physical address (Frame Number f, Offset d) via the Page Table.",
                        "Translation Lookaside Buffer (TLB): A high-speed associative hardware cache holding recent page-to-frame translations. Effective Memory Access Time (EMAT) = HitRate * (TLB_access + RAM_access) + (1 - HitRate) * (TLB_access + 2 * RAM_access).",
                    ]
                ),
                (
                    "2. Page Faults and Page Replacement Algorithms",
                    [
                        "When a process accesses a page not present in physical memory (valid-invalid bit set to invalid), a Page Fault interrupt traps to the OS kernel. The OS locates the page on backing store (swap disk), finds a free frame, reads the page into memory, updates the page table, and restarts the faulting instruction.",
                        "Algorithms for victim page selection:",
                        "• FIFO: Replaces the oldest page. Suffers from Belady's Anomaly (increasing frames can increase page faults).",
                        "• Optimal (OPT / Belady's Min): Replaces page that will not be referenced for longest future duration. Serves as a theoretical benchmark.",
                        "• Least Recently Used (LRU): Replaces page that has not been referenced for the longest period. Implemented via hardware timestamp registers or doubly linked stacks.",
                    ]
                )
            ]
        )
    ]

    # 3. CS103: Database Management Systems
    cs103_pages = [
        (
            "Page 1: Relational Model, Keys, and Relational Algebra",
            [
                (
                    "1. Foundations of the Relational Model",
                    [
                        "The relational model organizes data into Relations (tables) comprised of Tuples (rows) and Attributes (columns). Each attribute is constrained to an atomic domain of values.",
                        "Key Terminology:",
                        "• Superkey: A set of one or more attributes that uniquely identifies a tuple within a relation.",
                        "• Candidate Key: A minimal superkey (no proper subset is itself a superkey).",
                        "• Primary Key: The specific candidate key chosen by the database designer as the principal tuple identifier. Primary keys cannot contain NULL values (Entity Integrity).",
                        "• Foreign Key: An attribute set in a referencing relation whose values must match the primary key of a referenced relation or be NULL (Referential Integrity).",
                    ]
                ),
                (
                    "2. Relational Algebra and SQL Queries",
                    [
                        "Relational algebra provides formal theoretical foundation for relational databases: Select (sigma), Project (pi), Union, Set Difference, Cartesian Product, and Natural Join (bowtie).",
                        "SQL translates these operations declaratively. Join types include:",
                        "• Inner Join: Returns rows when matching values exist in both tables.",
                        "• Left Outer Join: Returns all rows from left table and matched rows from right table (NULL filled if no match).",
                        "• Right Outer Join and Full Outer Join: Symmetric outer join variants.",
                    ]
                )
            ]
        ),
        (
            "Page 2: Normalization and Functional Dependencies",
            [
                (
                    "1. Functional Dependencies and Modification Anomalies",
                    [
                        "Poor database design induces update anomalies, insertion anomalies, and deletion anomalies. A Functional Dependency (FD) X -> Y asserts that whenever two tuples agree on attributes X, they must agree on attributes Y.",
                        "Armstrong's Axioms: Reflexivity (if Y subset X, then X -> Y), Augmentation (if X -> Y, then XZ -> YZ), and Transitivity (if X -> Y and Y -> Z, then X -> Z).",
                    ]
                ),
                (
                    "2. Normal Forms: 1NF, 2NF, 3NF, and BCNF",
                    [
                        "• First Normal Form (1NF): All attribute domains contain only atomic, indivisible values; repeating groups and multi-valued attributes are eliminated.",
                        "• Second Normal Form (2NF): Relation is in 1NF and every non-prime attribute is fully functionally dependent on every candidate key (no partial dependencies).",
                        "• Third Normal Form (3NF): Relation is in 2NF and no non-prime attribute is transitively dependent on a candidate key (for every X -> A, either X is a superkey or A is a prime attribute).",
                        "• Boyce-Codd Normal Form (BCNF): Stricter variant of 3NF. For every non-trivial functional dependency X -> Y, the determinant X must be a superkey.",
                    ]
                )
            ]
        ),
        (
            "Page 3: Transactions and ACID Guarantees",
            [
                (
                    "1. Transaction Definition and States",
                    [
                        "A transaction is a logical unit of database processing consisting of one or more read and write operations. A transaction must transition through discrete states: Active -> Partially Committed -> Committed, or if errors occur, Failed -> Aborted (followed by rollback).",
                    ]
                ),
                (
                    "2. The ACID Properties",
                    [
                        "• Atomicity: The 'all-or-nothing' principle. Either all transaction operations execute successfully to completion, or the database is rolled back to its pre-transaction state via Undo logs.",
                        "• Consistency: A transaction must transform the database from one valid state to another valid state, satisfying all declared schema constraints, unique keys, and foreign keys.",
                        "• Isolation: Concurrent transactions must execute without mutual interference. Intermediate uncommitted changes remain invisible to concurrent operations.",
                        "• Durability: Once a transaction commits, its modifications are permanently recorded in non-volatile storage and survive subsequent system crashes (enforced by Write-Ahead Logging WAL).",
                    ]
                )
            ]
        ),
        (
            "Page 4: Concurrency Control and Storage Indexing",
            [
                (
                    "1. Concurrency Control and Two-Phase Locking (2PL)",
                    [
                        "Uncontrolled concurrent access causes anomalies: Dirty Reads (reading uncommitted data), Non-repeatable Reads (data modified between reads), and Phantom Reads (rows inserted/deleted matching a query predicate).",
                        "Two-Phase Locking (2PL) protocol ensures conflict serializability:",
                        "• Growing Phase: Transaction may acquire locks (Shared S or Exclusive X) but cannot release any lock.",
                        "• Shrinking Phase: Transaction may release locks but cannot acquire new locks.",
                        "Strict 2PL: All exclusive locks are held until transaction commits or aborts, preventing cascading rollbacks.",
                    ]
                ),
                (
                    "2. Database Indexing: B+ Trees and Hash Indexes",
                    [
                        "• B+ Tree Index: Balanced multi-way search tree where all data records or record pointers reside exclusively in leaf nodes. Leaves are doubly linked to support fast range queries. Internal nodes store search keys and pointers, yielding high fan-out and shallow tree height (typically 3 to 4 levels for millions of records). Search, insert, and delete take O(log n) disk I/O.",
                        "• Hash Index: Uses a hash function to map keys to bucket blocks. Provides O(1) point lookups, but does not support range scans or sorted ordering.",
                    ]
                )
            ]
        )
    ]

    # 4. CS104: Computer Networks and Protocols
    cs104_pages = [
        (
            "Page 1: Network Architectures and Layering Models",
            [
                (
                    "1. OSI 7-Layer Reference Model vs. TCP/IP Architecture",
                    [
                        "Network protocol suites are organized hierarchically into distinct functional layers.",
                        "The ISO/OSI Reference Model defines 7 layers:",
                        "1. Physical Layer: Bit transmission over physical media (voltage levels, optical pulses).",
                        "2. Data Link Layer: Frame transmission, physical MAC addressing, framing, and CRC error detection.",
                        "3. Network Layer: Host-to-host packet routing, logical IP addressing, and path determination.",
                        "4. Transport Layer: End-to-end process-to-process communication, reliability, flow and congestion control.",
                        "5. Session Layer: Dialog management, session establishment, and checkpointing.",
                        "6. Presentation Layer: Syntax and semantics negotiation, data encryption, and compression.",
                        "7. Application Layer: Network interface for user applications (HTTP, DNS, SMTP).",
                        "The TCP/IP Internet model condenses this into 4 layers: Network Interface (Link), Internet, Transport, and Application.",
                    ]
                )
            ]
        ),
        (
            "Page 2: Network Layer, IP Addressing, and Routing",
            [
                (
                    "1. IPv4 Addressing and Subnetting",
                    [
                        "An IPv4 address is a 32-bit integer formatted as four dotted-decimal octets (e.g., 192.168.1.1). Classless Inter-Domain Routing (CIDR) uses variable-length prefix notation (/prefix) to allocate IP blocks efficiently.",
                        "Subnet Mask: A bitmask dividing an IP into Network ID and Host ID. For a /24 prefix, mask is 255.255.255.0, offering 2^(32-24) - 2 = 254 usable host addresses (subtracting network address and broadcast address).",
                        "Address Resolution Protocol (ARP) dynamically resolves logical IP addresses to physical 48-bit Ethernet MAC addresses on a local subnet.",
                    ]
                ),
                (
                    "2. Routing Algorithms: Distance-Vector vs. Link-State",
                    [
                        "• Distance-Vector Routing (e.g., RIP): Routers iteratively share their entire routing table with immediate neighbors using the Bellman-Ford algorithm. Subject to the Count-to-Infinity problem, mitigated by Split Horizon and Poison Reverse.",
                        "• Link-State Routing (e.g., OSPF): Routers flood Link-State Advertisements (LSAs) across the entire network domain and independently compute shortest path trees using Dijkstra's algorithm. Avoids routing loops and converges rapidly.",
                    ]
                )
            ]
        ),
        (
            "Page 3: Transport Layer: TCP vs. UDP and Congestion Control",
            [
                (
                    "1. TCP vs. UDP Comparison",
                    [
                        "• UDP (User Datagram Protocol): Connectionless, lightweight, unacknowledged transport protocol with minimal 8-byte header overhead. Provides no guarantees for delivery, ordering, or duplicate protection. Ideal for latency-sensitive streaming, DNS queries, and real-time VoIP.",
                        "• TCP (Transmission Control Protocol): Connection-oriented, reliable byte-stream protocol. Implements connection setup via the Three-Way Handshake (SYN, SYN-ACK, ACK) and teardown via Four-Way Handshake (FIN, ACK, FIN, ACK).",
                    ]
                ),
                (
                    "2. TCP Reliability and Congestion Control",
                    [
                        "TCP ensures reliable transmission through sequence numbers, cumulative acknowledgments, and retransmission timers. Flow control is maintained through the Receive Window (rwnd) advertised in TCP headers to avoid buffer overflow on the receiver.",
                        "Congestion Control manages network bottleneck capacity via the Congestion Window (cwnd):",
                        "• Slow Start: cwnd starts at 1 MSS and doubles every RTT (exponential growth) until reaching ssthresh.",
                        "• Congestion Avoidance: cwnd increases linearly by 1 MSS per RTT (Additive Increase).",
                        "• Fast Retransmit and Fast Recovery: Upon receiving 3 duplicate ACKs, TCP infers packet loss, retransmits without waiting for timer expiry, and halves ssthresh (Multiplicative Decrease - AIMD).",
                    ]
                )
            ]
        ),
        (
            "Page 4: Application Layer Protocols: DNS, HTTP, and TLS",
            [
                (
                    "1. Domain Name System (DNS)",
                    [
                        "DNS is a distributed, hierarchical database mapping human-readable hostnames to IP addresses. The hierarchy comprises Root Servers, Top-Level Domain (TLD) servers (.com, .org), and Authoritative Name Servers.",
                        "Key DNS Record Types: A (IPv4 address), AAAA (IPv6 address), CNAME (canonical alias), MX (mail exchange server), and TXT (arbitrary text, SPF/DKIM verification).",
                    ]
                ),
                (
                    "2. HTTP Evolution and TLS Security",
                    [
                        "• HTTP/1.1: Introduced persistent TCP connections (Keep-Alive) and chunked transfer encoding, but suffers from Head-of-Line (HoL) blocking on single TCP streams.",
                        "• HTTP/2: Replaces textual format with binary framing. Multiplexes concurrent requests and responses over a single TCP connection, utilizes HPACK header compression, and supports server push.",
                        "• HTTP/3: Operates over QUIC (a UDP-based transport protocol), eliminating TCP Head-of-Line blocking across independent streams and supporting zero-RTT connection resumption.",
                        "• Transport Layer Security (TLS): Provides end-to-end encryption, server authentication via X.509 certificates, and message integrity via HMAC or AEAD ciphers.",
                    ]
                )
            ]
        )
    ]

    create_pdf(
        os.path.join(data_dir, "cs101_data_structures.pdf"),
        "CS101: Data Structures and Algorithms",
        cs101_pages,
    )
    create_pdf(
        os.path.join(data_dir, "cs102_operating_systems.pdf"),
        "CS102: Operating Systems Concepts",
        cs102_pages,
    )
    create_pdf(
        os.path.join(data_dir, "cs103_database_systems.pdf"),
        "CS103: Database Management Systems",
        cs103_pages,
    )
    create_pdf(
        os.path.join(data_dir, "cs104_computer_networks.pdf"),
        "CS104: Computer Networks and Protocols",
        cs104_pages,
    )
    print("[OK] All sample Computer Science course PDFs successfully generated in data/raw/")


if __name__ == "__main__":
    generate_all_documents()
