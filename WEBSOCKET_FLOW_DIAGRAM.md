# WebSocket Conversion Flow - Visual Diagrams

## Sequence Diagram: Complete Conversion Flow

```
User                Frontend              Backend              OCR Parser
 │                    │                     │                      │
 │ Click Convert       │                     │                      │
 ├───────────────────>│                     │                      │
 │                    │ POST /convert       │                      │
 │                    ├────────────────────>│                      │
 │                    │                     │ Create Task          │
 │                    │                     │ Start Thread         │
 │                    │ 200 OK              │                      │
 │                    │ {conversion_id}     │                      │
 │                    │<────────────────────┤                      │
 │                    │                     │                      │
 │                    │ WebSocket Connect   │                      │
 │                    ├────────────────────>│                      │
 │                    │ /ws/conversion/{id} │                      │
 │                    │                     │ Send Initial Status  │
 │                    │ {status: pending}   │                      │
 │                    │<────────────────────┤                      │
 │                    │                     │                      │
 │                    │                     │ Background Thread    │
 │                    │                     │ Update: processing   │
 │                    │ {progress: 10%}     │                      │
 │                    │<────────────────────┤                      │
 │ Show Progress      │                     │                      │
 │ Bar 10%            │                     │ Call OCR Parser      │
 │<───────────────────┤                     ├─────────────────────>│
 │                    │                     │                      │
 │                    │ {progress: 50%}     │ Processing...        │
 │                    │<────────────────────┤                      │
 │ Update Progress    │                     │                      │
 │ Bar 50%            │                     │                      │
 │<───────────────────┤                     │                      │
 │                    │                     │                      │
 │                    │                     │ Conversion Complete  │
 │                    │ {progress: 100%}    │<─────────────────────┤
 │                    │ {status: completed} │                      │
 │                    │<────────────────────┤                      │
 │ Update Progress    │                     │                      │
 │ Bar 100%           │                     │                      │
 │ Refresh List       │                     │                      │
 │<───────────────────┤                     │                      │
 │                    │ WebSocket Close     │                      │
 │                    ├────────────────────>│                      │
 │                    │                     │                      │
 │ View Converted     │                     │                      │
 │ Document           │                     │                      │
 │<───────────────────┤                     │                      │
```

## State Transition Diagram

```
                    ┌─────────────┐
                    │   PENDING   │
                    │ (Queued)    │
                    └──────┬──────┘
                           │
                           │ Start Processing
                           ▼
                    ┌─────────────┐
                    │ PROCESSING  │
                    │ (Converting)│
                    └──────┬──────┘
                           │
                ┌──────────┴──────────┐
                │                     │
         Success│                     │Error
                ▼                     ▼
        ┌─────────────┐        ┌─────────────┐
        │ COMPLETED   │        │   ERROR     │
        │ (Done)      │        │ (Failed)    │
        └─────────────┘        └─────────────┘
```

## Progress Bar Animation

```
Start Conversion
       │
       ▼
    ┌─────────────────────────────────────┐
    │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ 0%
    └─────────────────────────────────────┘
       │
       ▼ (10% - Starting)
    ┌─────────────────────────────────────┐
    │ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ 10%
    └─────────────────────────────────────┘
       │
       ▼ (50% - Processing)
    ┌─────────────────────────────────────┐
    │ ██████████████████░░░░░░░░░░░░░░░░░ │ 50%
    └─────────────────────────────────────┘
       │
       ▼ (100% - Complete)
    ┌─────────────────────────────────────┐
    │ ████████████████████████████████████ │ 100%
    └─────────────────────────────────────┘
       │
       ▼
   Refresh List
```

## Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    DocumentList                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ State:                                             │  │
│  │ - documents[]                                      │  │
│  │ - converting (filename)                            │  │
│  │ - conversionProgress {id: progress}                │  │
│  │ - webSockets {id: ws}                              │  │
│  └────────────────────────────────────────────────────┘  │
│                          │                                │
│         ┌────────────────┼────────────────┐               │
│         │                │                │               │
│         ▼                ▼                ▼               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│  │ DataTable  │  │ Progress   │  │ Action     │          │
│  │            │  │ Bar        │  │ Buttons    │          │
│  │ - Filename │  │            │  │            │          │
│  │ - Size     │  │ Shows %    │  │ - View     │          │
│  │ - Status   │  │ during     │  │ - Convert  │          │
│  │ - Progress │  │ conversion │  │            │          │
│  └────────────┘  └────────────┘  └────────────┘          │
│         │                │                │               │
│         └────────────────┼────────────────┘               │
│                          │                                │
│                          ▼                                │
│              handleConvert() method                       │
│                          │                                │
│         ┌────────────────┼────────────────┐               │
│         │                │                │               │
│         ▼                ▼                ▼               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│  │ Call       │  │ Connect    │  │ Update     │          │
│  │ Convert    │  │ WebSocket  │  │ Progress   │          │
│  │ Endpoint   │  │            │  │ State      │          │
│  └────────────┘  └────────────┘  └────────────┘          │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │      DocumentService            │
        │                                 │
        │ convertDocument()                │
        │ connectToConversionProgress()    │
        │ getConversionStatus()            │
        └─────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    ┌────────┐      ┌──────────┐      ┌──────────┐
    │ HTTP   │      │ WebSocket│      │ HTTP     │
    │ POST   │      │ Connect  │      │ GET      │
    │        │      │          │      │          │
    └────────┘      └──────────┘      └──────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
        ┌─────────────────────────────────┐
        │      Backend (FastAPI)          │
        │                                 │
        │ /convert endpoint               │
        │ /ws/conversion/{id} endpoint     │
        │ /conversion-status/{id} endpoint │
        └─────────────────────────────────┘
```

## Data Flow Diagram

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│ Frontend: handleConvert()            │
│ - Get filename                       │
│ - Call convertDocument()             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ HTTP POST /convert                  │
│ - filename: "document.pdf"          │
│ - prompt_mode: "prompt_layout_all"  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Backend: convert_document()          │
│ - Validate filename                 │
│ - Create conversion task            │
│ - Start background thread           │
│ - Return conversion_id              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Response: {conversion_id, status}   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Frontend: connectToConversionProgress│
│ - Create WebSocket connection       │
│ - Set up callbacks                  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ WebSocket: /ws/conversion/{id}      │
│ - Connection established            │
│ - Receive initial status            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Backend: Background Thread          │
│ - Update status: pending→processing │
│ - Broadcast progress: 10%           │
│ - Call OCR parser                   │
│ - Broadcast progress: 50%           │
│ - Continue processing               │
│ - Broadcast progress: 100%          │
│ - Update status: completed          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ WebSocket Messages                  │
│ - {progress: 10%, status: processing}
│ - {progress: 50%, status: processing}
│ - {progress: 100%, status: completed}
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Frontend: onMessage Callback        │
│ - Update progress state             │
│ - Update progress bar               │
│ - Check status                      │
│ - If completed: refresh list        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ UI Update                           │
│ - Progress bar: 0% → 100%           │
│ - Status badge: Pending → Converted │
│ - Action button: Convert → View     │
│ - Document list refreshed           │
└─────────────────────────────────────┘
```

## Error Handling Flow

```
Start Conversion
    │
    ▼
┌─────────────────────────────────────┐
│ Validation                          │
│ - Check filename                    │
│ - Check file exists                 │
└─────────────────────────────────────┘
    │
    ├─ Error ──────────────────────┐
    │                              │
    │                              ▼
    │                    ┌──────────────────┐
    │                    │ Return Error     │
    │                    │ HTTP 400/404/500 │
    │                    └──────────────────┘
    │                              │
    │                              ▼
    │                    ┌──────────────────┐
    │                    │ Frontend Error   │
    │                    │ Handler          │
    │                    │ Show Error Toast │
    │                    └──────────────────┘
    │
    └─ OK ─────────────────────────┐
                                   │
                                   ▼
                        ┌──────────────────┐
                        │ Start Conversion │
                        │ Background Task  │
                        └──────────────────┘
                                   │
                                   ▼
                        ┌──────────────────┐
                        │ Processing       │
                        │ OCR Parser       │
                        └──────────────────┘
                                   │
                        ┌──────────┴──────────┐
                        │                     │
                   Success│                   │Error
                        ▼                     ▼
                ┌──────────────┐      ┌──────────────┐
                │ Broadcast    │      │ Broadcast    │
                │ Completion   │      │ Error        │
                │ {status:     │      │ {status:     │
                │  completed}  │      │  error}      │
                └──────────────┘      └──────────────┘
                        │                     │
                        ▼                     ▼
                ┌──────────────┐      ┌──────────────┐
                │ Frontend     │      │ Frontend     │
                │ Refresh List │      │ Show Error   │
                │ Success Toast│      │ Error Toast  │
                └──────────────┘      └──────────────┘
```

## Timeline Example

```
Time    Event                           Progress    Status
────────────────────────────────────────────────────────────
0ms     User clicks Convert             0%          pending
10ms    POST /convert sent              0%          pending
50ms    Response received               0%          pending
60ms    WebSocket connected             0%          pending
70ms    Initial status received         0%          pending
100ms   Background thread starts        10%         processing
500ms   OCR parser processing           50%         processing
1000ms  OCR parser processing           75%         processing
2000ms  OCR parser completes            100%        processing
2010ms  Completion broadcast            100%        completed
2020ms  Frontend receives completion    100%        completed
2030ms  Document list refreshed         100%        completed
2040ms  UI updated                      100%        completed
```

## Concurrent Conversions

```
User 1: Convert doc1.pdf
    │
    ├─ Conversion ID: uuid-1
    │  ├─ WebSocket: /ws/conversion/uuid-1
    │  └─ Progress: 0% → 100%
    │
User 2: Convert doc2.pdf
    │
    ├─ Conversion ID: uuid-2
    │  ├─ WebSocket: /ws/conversion/uuid-2
    │  └─ Progress: 0% → 100%
    │
User 3: Convert doc3.pdf
    │
    └─ Conversion ID: uuid-3
       ├─ WebSocket: /ws/conversion/uuid-3
       └─ Progress: 0% → 100%

All running concurrently in separate threads!
```

