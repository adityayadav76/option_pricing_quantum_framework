import requests
import qiskit
from qiskit.result import Result
#from qiskit.result.counts import Counts
import numpy as np
import uuid
import datetime


def scale_counts(original_counts, target_shots):
    original_total = sum(original_counts.values())

    if original_total == 0:
        raise ValueError("Original counts sum to zero.")

    scaled_counts = {}
    remainders = {}

    # Step 1 — scale and floor
    for bitstring, count in original_counts.items():
        scaled = count * target_shots / original_total
        integer_part = int(scaled)
        scaled_counts[bitstring] = integer_part
        remainders[bitstring] = scaled - integer_part

    # Step 2 — distribute remainder shots
    total_scaled = sum(scaled_counts.values())
    remaining = target_shots - total_scaled

    if remaining > 0:
        sorted_keys = sorted(
            remainders,
            key=remainders.get,
            reverse=True
        )

        for i in range(remaining):
            scaled_counts[sorted_keys[i]] += 1

    return scaled_counts



def _get_qindex(circ, name, index):
    """
    Find the qubit index.

    Args:
        circ: The Qiskit QuantumCircuit in question
        name: The name of the quantum register
        index: The qubit's relative index inside the register

    Returns:
        The qubit's absolute index if all registers are concatenated.
    """
    ret = 0
    for reg in circ.qregs:
        if name != reg.name:
            ret += reg.size
        else:
            return ret + index
    return ret + index
    
class AutomatskiKomencoQiskit:
    
    def __init__(self, host, port, apiKey: str = "open"):
        self.apiKey = apiKey
        self.host = host
        self.port = port
        self.gateMap = {}
        
        self.gateMap["cu1"]="cp"
        self.gateMap["cnot"]="cx"
        self.gateMap["phase"] = "p"
        self.gateMap["cphase"] = "cp"
        self.gateMap["ccphase"] = "ccp"
        self.gateMap["mcphase"] = "mcp"
        self.gateMap["toffoli"] = "ccx"
        self.gateMap["fredkin"] = "cswap"
        self.gateMap["cxp"] = "cnotp"
        self.gateMap["ccxp"] = "ccnotp"
        
    def run(self, circuit, repetitions=1000, topK=20, silent=False):
        self.silent = silent
        tstart = datetime.datetime.now()
        
        body = self.serialize_circuit(circuit, repetitions, topK)
        response = requests.post(f'http://{self.host}:{self.port}/api/komenco', json=body, timeout=None)
        response.close() #bug fix - too many connections open
        struct = response.json()
        
        tend = datetime.datetime.now()
        execution_time = (tend - tstart).microseconds
        if not self.silent: print(f"Time Taken {(tend - tstart)}")
        
        cannotProceed = None
        try:
            if struct["error"] :
                print(struct["error"])
                cannotProceed = True
                raise Exception(struct["error"])
        except Exception as e:
            if cannotProceed:
                raise e            
            pass 
        
        
        
        return self.deserialize_result(struct, repetitions, execution_time)

    def serialize_circuit(self, circuit, repetitions, topK):
        # Extract the number of qubits
        num_qubits = circuit.num_qubits
        
        operations = []
        measurements = []

        for instr, qargs, cargs in circuit.data:
            gate = instr.name.lower() #the server uses lowercase gate names 
            
            #we need to map some gate naming conventions to make sure there are no errors
            if gate in self.gateMap:
                gate = self.gateMap[gate]
                   
            params = [param for param in instr.params]
            #print([q for q in qargs])
            qubits = []
            # Get qbit arguments
            # ref: myqlm converters [interop]
            for qarg in qargs:
                if qarg._register :
                    qubits.append(
                        _get_qindex(circuit, qarg._register.name, qarg._index))
                else:
                    qubit_index = circuit.find_bit(qarg).index
                    qubits.append(qubit_index)
                    
            if gate == 'measure':
                for q in qubits:
                    measurements.append(q)
            elif gate in ['barrier']:
                continue 
            elif gate in ['delay','initialize','store']: # removed 'reset'
                raise(Exception(f"gate or operation: '{gate}' is not supported by Komenco yet"))                
            else:
                operations.append({"gate": gate, "params": params, "qubits": qubits})
        
        if not self.silent: print("Executing Quantum Circuit With...")
        if not self.silent: print(f"{num_qubits} Qubits And ...")
        if not self.silent: print(f"{len(operations)} Gates")
        
        if len(measurements) == 0:
            raise(Exception("There are no measurements done at the end of the circuit."))        
                
        return { "num_qubits": num_qubits, "operations": operations, "measurements": measurements, "shots": repetitions, "topK": topK, "apiKey":self.apiKey}    

    @staticmethod
    def deserialize_result(response_data, repetitions, execution_time):
        measurementsStrings = response_data['measurements']
        # Build counts and probabilities
        counts = {}
        probabilities = {}
        for key_original, value in measurementsStrings.items():
            #combo = np.array( [int(char) for char in key] , dtype=np.int32)
            count = round(value*repetitions)
            if count == 0:
                continue            
            #key = hex(int(key_original, 2)) # key_original[::-1]
            key = key_original 
            counts[key] = count
            probabilities[key] = value
        
        # this method is used to make sure the total number of counts matches
        # the repetitions
        counts = scale_counts(counts, repetitions)
        
        job_result = [
            {
                "data": {
                    "counts": counts,
                    "probabilities": probabilities,
                    # Qiskit/experiments relies on this being present in this location in the
                    # ExperimentData class.
                    "metadata": {},
                },
                "shots": repetitions,
                "header": {},
                "success": True,
            }
        ]
        
        
        return Result.from_dict(
            {
                "results": job_result,
                "job_id": str(uuid.uuid4()),
                "backend_name": "Automatski Yocto",
                "backend_version": "V1.0b",
                "qobj_id": str(uuid.uuid4()),
                "success": True,
                "time_taken": execution_time,
            }
        ) 
        
        
import uuid
import datetime
from typing import List, Union

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2, Options, JobV1
from qiskit.providers.jobstatus import JobStatus
from qiskit.result import Result
from qiskit.transpiler import Target
from qiskit.circuit.library import (
    XGate, YGate, ZGate, HGate, PhaseGate,
    RXGate, RYGate, RZGate,
    CXGate, CZGate, CCXGate, SwapGate,
    SGate, SdgGate, TGate, TdgGate,
    SXGate, SXdgGate, IGate
)
from qiskit.circuit import Measure, Reset
from qiskit import transpile

# ============================================================
# Custom Job Object
# ============================================================

class AutomatskiJob(JobV1):

    def __init__(self, backend, job_id, result):
        super().__init__(backend, job_id)
        self._result = result
        self._status = JobStatus.DONE

    def submit(self):
        # Since execution already happened in backend.run(),
        # nothing to do here.
        pass

    def result(self):
        return self._result

    def status(self):
        return self._status


# ============================================================
# Automatski Backend
# ============================================================

class AutomatskiBackendV2(BackendV2):

    def __init__(self, num_qubits: int, thebackend, apiKey: str = "open"):
        self.apiKey = apiKey
        self._num_qubits = num_qubits
        self._target = self._build_target(num_qubits)
        self.thebackend = thebackend
        
        super().__init__(
            name="automatski_backend",
            description="Automatski Quantum Computer Backend",
            backend_version="1.0",
            online_date=datetime.datetime.now(),
        )

    # --------------------------------------------------------
    # Required: Target
    # --------------------------------------------------------
    @property
    def target(self):
        return self._target

    def _build_target(self, num_qubits):
        target = Target(num_qubits=num_qubits)

        # Single qubit gates
        single_qubit_gates = [
            XGate(), YGate(), ZGate(), HGate(), PhaseGate(0),
            RXGate(0), RYGate(0), RZGate(0),
            SGate(), SdgGate(), TGate(), TdgGate(),
            SXGate(), SXdgGate(), IGate()
        ]

        for gate in single_qubit_gates:
            target.add_instruction(gate, {(i,): None for i in range(num_qubits)})

        # Two qubit gates (fully connected)
        two_qubit_gates = [CXGate(), CZGate(), SwapGate()]
        for gate in two_qubit_gates:
            target.add_instruction(
                gate,
                {(i, j): None for i in range(num_qubits)
                             for j in range(num_qubits) if i != j}
            )
        
        '''
        # Three qubit
        target.add_instruction(
            CCXGate(),
            {(i, j, k): None
             for i in range(num_qubits)
             for j in range(num_qubits)
             for k in range(num_qubits)
             if len({i, j, k}) == 3}
        )
        '''
        
        # Measurement
        target.add_instruction(
            Measure(),
            {(i,): None for i in range(num_qubits)}
        )
        #Reset
        target.add_instruction(
            Reset(),
            {(i,): None for i in range(num_qubits)}
        )
        return target

    # --------------------------------------------------------
    # Required: max_circuits
    # --------------------------------------------------------
    @property
    def max_circuits(self):
        return None  # unlimited

    # --------------------------------------------------------
    # Required: Default Options
    # --------------------------------------------------------
    @classmethod
    def _default_options(cls):
        return Options(
            shots=100000,
        )

    def set_options(self, **fields):
        run_opts = self.options
        #run_opts = self.options.__class__(**self.options.__dict__)
        run_opts.update_options(**fields)


    # --------------------------------------------------------
    # Required: Run
    # --------------------------------------------------------
    def run(self, run_input: QuantumCircuit | list[QuantumCircuit], parameter_binds=None, **options):

        # Start from backend default options
        run_opts = self.options

        # Override with runtime options (if provided)
        if options:
            #run_opts = self.options.__class__(**self.options.__dict__)
            run_opts.update_options(**options)

        shots = run_opts.shots

        if not isinstance(run_input, list):
            run_input = [run_input]

        bound_circuits = []

        for i, circuit in enumerate(run_input):

            if circuit.parameters:
                if parameter_binds is None:
                    raise ValueError("Circuit has unbound parameters but no parameter_binds were supplied.")

                bind = parameter_binds[i]
                circuit = circuit.assign_parameters(bind)

            bound_circuits.append(circuit)

        results = []

        for circuit in bound_circuits:

            # --------------------------------------------------
            # HERE you call your internal simulator
            # --------------------------------------------------
            counts = self._execute_circuit(
                circuit,
                shots=shots
            )
            
            #print(counts)

            num_bytes = (circuit.num_clbits + 7) // 8
            memory = []
            for bitstring, count in counts.items():
                for _ in range(count):
                    # Convert bitstring to integer
                    value = int(bitstring, 2)
                    # Convert to zero-padded hex without 0x
                    hex_string = format(value, f'0{num_bytes*2}x')
                    memory.append(hex_string)
            # this is needed to fix the problem when the number of shots is not 
            # equal to the total number of counts
            if len(memory) > shots:
                memory = memory[:shots]
            elif len(memory) < shots:
                # pad with last valid sample
                last = memory[-1] if memory else "00"
                memory.extend([last] * (shots - len(memory)))
            
            # this method is used to make sure the total number of counts matches
            # the shots
            counts = scale_counts(counts, shots)
            
            
            results.append({
                "success": True,
                "shots": shots,
                "data": {
                    "counts": counts,
                    "memory": memory
                },
                "header": {
                    "name": circuit.name
                }
            })

        result_obj = Result.from_dict({
            "backend_name": self.name,
            "backend_version": self.backend_version,
            "qobj_id": None,
            "job_id": str(uuid.uuid4()),
            "success": True,
            "results": results
        })

        return AutomatskiJob(self, result_obj.job_id, result_obj)

    # --------------------------------------------------------
    # The internal execution engine
    # --------------------------------------------------------
    def _execute_circuit(self, circuit, shots):
        """
        Convert Qiskit circuit → Automatski IR → execute → return counts dict
        Must return:
            { '000': 512, '111': 512 }
        """
        qiskit_qc = transpile(
            circuit,
            basis_gates=[
                'ccx','ccz','cp','crz','cs','csdg','cswap','cu',
                'cx','cy','cz','h','id','measure','reset', 'p','rx','ry','rz',
                's','sdg','swap','sx','sxdg','t','tdg','u','x','y','z','p'
            ],
            optimization_level=3
        )

        result_sim = self.thebackend.run(qiskit_qc, repetitions=shots, topK=100, silent=True)
        result_dic = result_sim.get_counts(None)
        
        return result_dic        
        
