// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2022-2023 SUSE LLC
//
// Author: Joerg Roedel <jroedel@suse.de>

use crate::address::{Address, PhysAddr, VirtAddr};
use crate::cpu::msr::{SEV_GHCB, write_msr};
use crate::cpu::percpu::{this_cpu, with_pgtable_mut};
use crate::cpu::{IrqGuard, X86GeneralRegs, flush_tlb_global_sync_page};
use crate::error::SvsmError;
use crate::mm::validate::{
    valid_bitmap_clear_valid_4k, valid_bitmap_set_valid_4k, valid_bitmap_valid_addr,
};
use crate::mm::virt_to_phys;
use crate::platform::PageStateChangeOp;
use crate::sev::hv_doorbell::HVDoorbell;
use crate::sev::utils::raw_vmgexit;
use crate::types::{Bytes, GUEST_VMPL, PAGE_SIZE_2M, PageSize};
use crate::utils::MemoryRegion;

use crate::mm::PageBox;
use core::arch::global_asm;
use core::arch::x86_64::CpuidResult;
use core::mem::{self, MaybeUninit, offset_of};
use core::ops::Deref;
use core::ptr;
use core::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};

use super::msr_protocol::{invalidate_page_msr, register_ghcb_gpa_msr, validate_page_msr};
use super::{PvalidateOp, pvalidate};

use zerocopy::{FromZeros, Immutable, IntoBytes};

#[repr(C, packed)]
#[derive(Debug, Default, Clone, Copy, IntoBytes, Immutable)]
pub struct PageStateChangeHeader {
    cur_entry: u16,
    end_entry: u16,
    reserved: u32,
}

const PSC_GFN_MASK: u64 = ((1u64 << 52) - 1) & !0xfffu64;

const PSC_OP_SHIFT: u8 = 52;
const PSC_OP_PRIVATE: u64 = 1 << PSC_OP_SHIFT;
const PSC_OP_SHARED: u64 = 2 << PSC_OP_SHIFT;
const PSC_OP_PSMASH: u64 = 3 << PSC_OP_SHIFT;
const PSC_OP_UNSMASH: u64 = 4 << PSC_OP_SHIFT;

const PSC_FLAG_HUGE_SHIFT: u8 = 56;
const PSC_FLAG_HUGE: u64 = 1 << PSC_FLAG_HUGE_SHIFT;

const GHCB_BUFFER_SIZE: usize = 0x7f0;

macro_rules! ghcb_getter {
    ($name:ident, $field:ident,$t:ty) => {
        #[allow(unused)]
        fn $name(&self) -> Result<$t, GhcbError> {
            self.is_valid(offset_of!(Self, $field))
                .then(|| self.$field.load(Ordering::Relaxed))
                .ok_or(GhcbError::VmgexitInvalid)
        }
    };
}

macro_rules! ghcb_setter {
    ($name:ident, $field:ident, $t:ty) => {
        #[allow(unused)]
        fn $name(&self, val: $t) {
            self.$field.store(val, Ordering::Relaxed);
            self.set_valid(offset_of!(Self, $field));
        }
    };
}

#[derive(Clone, Copy, Debug)]
pub enum GhcbError {
    // Attempted to write at an invalid offset in the GHCB
    InvalidOffset,
    // A response from the hypervisor after VMGEXIT is invalid
    VmgexitInvalid,
    // A response from the hypervisor included an error code
    VmgexitError(u64, u64),
}

impl From<GhcbError> for SvsmError {
    fn from(e: GhcbError) -> Self {
        Self::Ghcb(e)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
#[expect(non_camel_case_types, clippy::upper_case_acronyms)]
enum GHCBExitCode {
    RDTSC = 0x6e,
    CPUID = 0x72,
    IOIO = 0x7b,
    MSR = 0x7c,
    VMMCALL = 0x81,
    RDTSCP = 0x87,
    MMIO_READ = 0x8000_0001,
    MMIO_WRITE = 0x8000_0002,
    SNP_PSC = 0x8000_0010,
    GUEST_REQUEST = 0x8000_0011,
    GUEST_EXT_REQUEST = 0x8000_0012,
    AP_CREATE = 0x80000013,
    HV_DOORBELL = 0x8000_0014,
    HV_IPI = 0x8000_0015,
    CONFIGURE_INT_INJ = 0x8000_001B,
    DISABLE_ALT_INJ = 0x8000_001C,
    SPECIFIC_EOI = 0x8000_001D,
}

#[derive(Clone, Copy, Debug)]
pub enum GHCBIOSize {
    Size8,
    Size16,
    Size32,
}

impl TryFrom<Bytes> for GHCBIOSize {
    type Error = SvsmError;

    fn try_from(size: Bytes) -> Result<GHCBIOSize, Self::Error> {
        match size {
            Bytes::One => Ok(GHCBIOSize::Size8),
            Bytes::Two => Ok(GHCBIOSize::Size16),
            Bytes::Four => Ok(GHCBIOSize::Size32),
            _ => Err(SvsmError::InvalidBytes),
        }
    }
}

#[derive(Debug)]
pub struct GhcbPage(PageBox<GHCB>);

impl GhcbPage {
    pub fn new() -> Result<Self, SvsmError> {
        let page = PageBox::<GHCB>::try_new_zeroed()?;
        let vaddr = page.vaddr();
        let paddr = virt_to_phys(vaddr);

        // SAFETY: we trust virt_to_phys() to return GHCB's physical address
        // since it panics if vaddr is invalid.
        unsafe {
            // Make page invalid
            pvalidate(vaddr, PageSize::Regular, PvalidateOp::Invalid)?;

            // Let the Hypervisor take the page back
            invalidate_page_msr(paddr)?;
        }

        // Needs guarding for Stage2 GHCB
        if valid_bitmap_valid_addr(paddr) {
            valid_bitmap_clear_valid_4k(paddr);
        }

        // Map page unencrypted
        with_pgtable_mut(|pg| pg.set_shared_4k(vaddr))?;
        flush_tlb_global_sync_page(vaddr, PageSize::Regular);

        // SAFETY: all zeros is a valid representation for the GHCB.
        Ok(Self(page))
    }
}

impl Drop for GhcbPage {
    fn drop(&mut self) {
        let vaddr = self.0.vaddr();
        let paddr = virt_to_phys(vaddr);

        // Re-encrypt page
        with_pgtable_mut(|pg| pg.set_encrypted_4k(vaddr)).expect("Could not re-encrypt page");
        flush_tlb_global_sync_page(vaddr, PageSize::Regular);

        // Unregister GHCB PA
        // SAFETY: mapping the GHCB at physical address 0 is safe.
        unsafe {
            register_ghcb_gpa_msr(PhysAddr::null()).expect("Could not unregister GHCB");
        }

        // SAFETY: we trust virt_to_phys() to return GHCB's physical address
        // since it panics if vaddr is invalid.
        unsafe {
            // Ask the hypervisor to change the page back to the private page
            // state.
            validate_page_msr(paddr).expect("Could not change page state");

            // Make page guest-valid
            pvalidate(vaddr, PageSize::Regular, PvalidateOp::Valid)
                .expect("Could not pvalidate page");
        }

        // Needs guarding for Stage2 GHCB
        if valid_bitmap_valid_addr(paddr) {
            valid_bitmap_set_valid_4k(paddr);
        }
    }
}

impl Deref for GhcbPage {
    type Target = GHCB;
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

#[repr(C)]
#[derive(Debug, FromZeros)]
pub struct GHCB {
    reserved_1: [AtomicU8; 0xcb],
    cpl: AtomicU8,
    reserved_2: [AtomicU8; 0x74],
    xss: AtomicU64,
    reserved_3: [AtomicU8; 0x18],
    dr7: AtomicU64,
    reserved_4: [AtomicU8; 0x90],
    rax: AtomicU64,
    reserved_5: [AtomicU8; 0x100],
    reserved_6: AtomicU64,
    rcx: AtomicU64,
    rdx: AtomicU64,
    rbx: AtomicU64,
    reserved_7: AtomicU64,
    rbp: AtomicU64,
    rsi: AtomicU64,
    rdi: AtomicU64,
    r8: AtomicU64,
    r9: AtomicU64,
    r10: AtomicU64,
    r11: AtomicU64,
    r12: AtomicU64,
    r13: AtomicU64,
    r14: AtomicU64,
    r15: AtomicU64,
    reserved_8: [AtomicU8; 0x10],
    sw_exit_code: AtomicU64,
    sw_exit_info_1: AtomicU64,
    sw_exit_info_2: AtomicU64,
    sw_scratch: AtomicU64,
    reserved_9: [AtomicU8; 0x38],
    xcr0: AtomicU64,
    valid_bitmap: [AtomicU64; 2],
    x87_state_gpa: AtomicU64,
    reserved_10: [AtomicU8; 0x3f8],
    buffer: [AtomicU8; GHCB_BUFFER_SIZE],
    reserved_11: [AtomicU8; 0xa],
    version: AtomicU16,
    usage: AtomicU32,
}

impl GHCB {
    ghcb_getter!(get_cpl_valid, cpl, u8);
    ghcb_setter!(set_cpl_valid, cpl, u8);

    ghcb_getter!(get_xss_valid, xss, u64);
    ghcb_setter!(set_xss_valid, xss, u64);

    ghcb_getter!(get_rax_valid, rax, u64);
    ghcb_setter!(set_rax_valid, rax, u64);

    ghcb_getter!(get_rcx_valid, rcx, u64);
    ghcb_setter!(set_rcx_valid, rcx, u64);

    ghcb_getter!(get_rdx_valid, rdx, u64);
    ghcb_setter!(set_rdx_valid, rdx, u64);

    ghcb_getter!(get_rbx_valid, rbx, u64);
    ghcb_setter!(set_rbx_valid, rbx, u64);

    ghcb_getter!(get_r8_valid, r8, u64);
    ghcb_setter!(set_r8_valid, r8, u64);

    ghcb_getter!(get_exit_code_valid, sw_exit_code, u64);
    ghcb_setter!(set_exit_code_valid, sw_exit_code, u64);

    ghcb_getter!(get_exit_info_1_valid, sw_exit_info_1, u64);
    ghcb_setter!(set_exit_info_1_valid, sw_exit_info_1, u64);

    ghcb_getter!(get_exit_info_2_valid, sw_exit_info_2, u64);
    ghcb_setter!(set_exit_info_2_valid, sw_exit_info_2, u64);

    ghcb_getter!(get_sw_scratch_valid, sw_scratch, u64);
    ghcb_setter!(set_sw_scratch_valid, sw_scratch, u64);

    ghcb_getter!(get_sw_xcr0_valid, xcr0, u64);
    ghcb_setter!(set_sw_xcr0_valid, xcr0, u64);

    ghcb_getter!(get_sw_x87_state_gpa_valid, x87_state_gpa, u64);
    ghcb_setter!(set_sw_x87_state_gpa_valid, x87_state_gpa, u64);

    ghcb_getter!(get_version_valid, version, u16);
    ghcb_setter!(set_version_valid, version, u16);

    ghcb_getter!(get_usage_valid, usage, u32);
    ghcb_setter!(set_usage_valid, usage, u32);

    pub fn vmmcall(&self, regs: &mut X86GeneralRegs) -> Result<(), SvsmError> {
        self.clear();
        self.set_rcx_valid(regs.rcx as u64);
        self.set_rdx_valid(regs.rdx as u64);
        self.set_r8_valid(regs.r8 as u64);
        self.vmgexit(GHCBExitCode::VMMCALL, 0, 0)?;
        regs.rax = self.get_rax_valid()? as usize;
        Ok(())
    }

    pub fn cpuid(&self, eax: u32, ecx: u32) -> Result<CpuidResult, SvsmError> {
        self.clear();
        self.set_rax_valid(eax as u64);
        self.set_rcx_valid(ecx as u64);
        self.vmgexit(GHCBExitCode::CPUID, 0, 0)?;
        Ok(CpuidResult {
            eax: self.get_rax_valid()? as u32,
            ebx: self.get_rbx_valid()? as u32,
            ecx: self.get_rcx_valid()? as u32,
            edx: self.get_rdx_valid()? as u32,
        })
    }

    pub fn cpuid_regs(&self, regs: &mut X86GeneralRegs) -> Result<(), SvsmError> {
        let result = self.cpuid(regs.rax as u32, regs.rcx as u32)?;
        regs.rax = result.eax as usize;
        regs.rbx = result.ebx as usize;
        regs.rcx = result.ecx as usize;
        regs.rdx = result.edx as usize;
        Ok(())
    }

    pub fn rdtscp_regs(&self, regs: &mut X86GeneralRegs) -> Result<(), SvsmError> {
        self.clear();
        self.vmgexit(GHCBExitCode::RDTSCP, 0, 0)?;
        let rax = self.get_rax_valid()?;
        let rdx = self.get_rdx_valid()?;
        let rcx = self.get_rcx_valid()?;
        regs.rax = rax as usize;
        regs.rdx = rdx as usize;
        regs.rcx = rcx as usize;
        Ok(())
    }

    pub fn rdtsc_regs(&self, regs: &mut X86GeneralRegs) -> Result<(), SvsmError> {
        self.clear();
        self.vmgexit(GHCBExitCode::RDTSC, 0, 0)?;
        let rax = self.get_rax_valid()?;
        let rdx = self.get_rdx_valid()?;
        regs.rax = rax as usize;
        regs.rdx = rdx as usize;
        Ok(())
    }

    pub fn wrmsr(&self, msr_index: u32, value: u64) -> Result<(), SvsmError> {
        self.wrmsr_raw(msr_index as u64, value & 0xFFFF_FFFF, value >> 32)
    }

    pub fn wrmsr_regs(&self, regs: &X86GeneralRegs) -> Result<(), SvsmError> {
        self.wrmsr_raw(regs.rcx as u64, regs.rax as u64, regs.rdx as u64)
    }

    pub fn wrmsr_raw(&self, rcx: u64, rax: u64, rdx: u64) -> Result<(), SvsmError> {
        self.clear();

        self.set_rcx_valid(rcx);
        self.set_rax_valid(rax);
        self.set_rdx_valid(rdx);

        self.vmgexit(GHCBExitCode::MSR, 1, 0)?;
        Ok(())
    }

    /// Perform an MSR read via a GHCB call
    ///
    /// # Arguments
    ///
    /// - `msr_index` - MSR to read
    ///
    /// # Returns
    ///
    /// On a successful call, returns `Result` with a tuple `(u32, u32)`. The
    /// first value is the `RAX` and the second the `RDX` register. On failure
    /// a `Result` with `SvsmError` is returned.
    pub fn rdmsr_raw(&self, msr_index: u32) -> Result<(u32, u32), SvsmError> {
        self.clear();
        self.set_rcx_valid(msr_index as u64);
        self.vmgexit(GHCBExitCode::MSR, 0, 0)?;
        let eax: u32 = (self.get_rax_valid()? & 0xFFFFFFFF) as u32;
        let edx: u32 = (self.get_rdx_valid()? & 0xFFFFFFFF) as u32;
        Ok((eax, edx))
    }

    /// Perform an MSR read via a GHCB call an return a `u64` value.
    ///
    /// # Arguments
    ///
    /// - `msr_index` - MSR to read
    ///
    /// # Returns
    ///
    /// On a successful call, returns `Result` with a `u64` value. On failure
    /// a `Result` with `SvsmError` is returned.
    pub fn rdmsr(&self, msr_index: u32) -> Result<u64, SvsmError> {
        let (rax, rdx) = self.rdmsr_raw(msr_index)?;
        Ok((rax as u64) | ((rdx as u64) << 32))
    }

    pub fn rdmsr_regs(&self, regs: &mut X86GeneralRegs) -> Result<(), SvsmError> {
        let (rax, rdx) = self.rdmsr_raw(u32::try_from(regs.rcx).unwrap())?;
        regs.rdx = rdx as usize;
        regs.rax = rax as usize;
        Ok(())
    }

    pub fn register(&self) -> Result<(), SvsmError> {
        let vaddr = VirtAddr::from(self as *const GHCB);
        let paddr = virt_to_phys(vaddr);

        // Register GHCB GPA
        // SAFETY: we trust virt_to_phys() to return self's physical address
        // since it panics if vaddr is invalid.
        unsafe { Ok(register_ghcb_gpa_msr(paddr)?) }
    }

    pub fn clear(&self) {
        // Clear valid bitmap
        self.valid_bitmap[0].store(0, Ordering::Relaxed);
        self.valid_bitmap[1].store(0, Ordering::Relaxed);

        // Mark valid_bitmap valid
        let off = offset_of!(Self, valid_bitmap);
        self.set_valid(off);
        self.set_valid(off + mem::size_of::<u64>());
    }

    fn set_valid(&self, offset: usize) {
        let bit: usize = (offset >> 3) & 0x3f;
        let index: usize = (offset >> 9) & 0x1;
        let mask: u64 = 1 << bit;

        self.valid_bitmap[index].fetch_or(mask, Ordering::Relaxed);
    }

    fn is_valid(&self, offset: usize) -> bool {
        let bit: usize = (offset >> 3) & 0x3f;
        let index: usize = (offset >> 9) & 0x1;
        let mask: u64 = 1 << bit;

        (self.valid_bitmap[index].load(Ordering::Relaxed) & mask) == mask
    }

    fn vmgexit(
        &self,
        exit_code: GHCBExitCode,
        exit_info_1: u64,
        exit_info_2: u64,
    ) -> Result<(), GhcbError> {
        // GHCB is version 2
        self.set_version_valid(2);
        // GHCB Follows standard format
        self.set_usage_valid(0);
        self.set_exit_code_valid(exit_code as u64);
        self.set_exit_info_1_valid(exit_info_1);
        self.set_exit_info_2_valid(exit_info_2);

        // Test-only seam: when a mock hypervisor is installed, skip MSR/VMGEXIT
        // hardware and let the mock mutate the GHCB (including the shared buffer)
        // in place. Production logic below (SW_EXITINFO1 check) still runs unchanged.
        #[cfg(test)]
        if psc_repro_mock::invoke_if_installed(self) {
            // Fall through to the existing SW_EXITINFO1 check.
        } else {
            let ghcb_address = VirtAddr::from(self as *const GHCB);
            let ghcb_pa = u64::from(virt_to_phys(ghcb_address));
            // Disable interrupts between writing the MSR and making the GHCB call
            // to prevent reentrant use of the GHCB MSR.
            let guard = IrqGuard::new();
            // SAFETY: GHCB is already allocated and setup so this is safe.
            unsafe {
                write_msr(SEV_GHCB, ghcb_pa);
            }
            raw_vmgexit();
            drop(guard);
        }

        #[cfg(not(test))]
        {
            let ghcb_address = VirtAddr::from(self as *const GHCB);
            let ghcb_pa = u64::from(virt_to_phys(ghcb_address));
            // Disable interrupts between writing the MSR and making the GHCB call
            // to prevent reentrant use of the GHCB MSR.
            let guard = IrqGuard::new();
            // SAFETY: GHCB is already allocated and setup so this is safe.
            unsafe {
                write_msr(SEV_GHCB, ghcb_pa);
            }
            raw_vmgexit();
            drop(guard);
        }

        let sw_exit_info_1 = self.get_exit_info_1_valid()?;
        if sw_exit_info_1 != 0 {
            return Err(GhcbError::VmgexitError(
                sw_exit_info_1,
                self.sw_exit_info_2.load(Ordering::Relaxed),
            ));
        }

        Ok(())
    }

    pub fn ioio_in(&self, port: u16, size: GHCBIOSize) -> Result<u64, SvsmError> {
        self.clear();

        let mut info: u64 = 1; // IN instruction

        info |= (port as u64) << 16;

        match size {
            GHCBIOSize::Size8 => info |= 1 << 4,
            GHCBIOSize::Size16 => info |= 1 << 5,
            GHCBIOSize::Size32 => info |= 1 << 6,
        }

        self.vmgexit(GHCBExitCode::IOIO, info, 0)?;
        let rax = self.get_rax_valid()?;
        Ok(rax)
    }

    pub fn ioio_out(&self, port: u16, size: GHCBIOSize, value: u64) -> Result<(), SvsmError> {
        self.clear();

        let mut info: u64 = 0; // OUT instruction

        info |= (port as u64) << 16;

        match size {
            GHCBIOSize::Size8 => info |= 1 << 4,
            GHCBIOSize::Size16 => info |= 1 << 5,
            GHCBIOSize::Size32 => info |= 1 << 6,
        }

        self.set_rax_valid(value);
        self.vmgexit(GHCBExitCode::IOIO, info, 0)?;
        Ok(())
    }

    fn write_buffer<T>(&self, data: &T, offset: usize) -> Result<(), GhcbError>
    where
        T: IntoBytes + Immutable,
    {
        let src = data.as_bytes();
        self.write_buffer_slice(src, offset)
    }

    fn write_buffer_slice(&self, data: &[u8], offset: usize) -> Result<(), GhcbError> {
        let dst = &self
            .buffer
            .get(offset..)
            .ok_or(GhcbError::InvalidOffset)?
            .get(..data.len())
            .ok_or(GhcbError::InvalidOffset)?;
        for (dst, src) in dst.iter().zip(data.iter().copied()) {
            dst.store(src, Ordering::Relaxed);
        }
        Ok(())
    }

    fn read_buffer_slice(
        &self,
        data: &mut [MaybeUninit<u8>],
        offset: usize,
    ) -> Result<(), GhcbError> {
        let src = &self
            .buffer
            .get(offset..)
            .ok_or(GhcbError::InvalidOffset)?
            .get(..data.len())
            .ok_or(GhcbError::InvalidOffset)?;
        for (d, s) in data.iter_mut().zip(src.iter()) {
            d.write(s.load(Ordering::Relaxed));
        }
        Ok(())
    }

    fn pepare_buffer(&self) {
        let buffer_va = VirtAddr::from(self.buffer.as_ptr());
        let buffer_pa = u64::from(virt_to_phys(buffer_va));
        self.set_sw_scratch_valid(buffer_pa);
    }

    /// Perfrom a write to a memory-mapped IO area
    ///
    /// # Safety
    ///
    /// Caller must ensure that `pa` points to a properly aligned memory location and the
    /// memory accessed is part of a valid MMIO range.
    pub unsafe fn mmio_write(&self, pa: PhysAddr, value: &[u8]) -> Result<(), SvsmError> {
        self.clear();
        self.write_buffer_slice(value, 0)?;
        self.pepare_buffer();
        self.vmgexit(GHCBExitCode::MMIO_WRITE, u64::from(pa), value.len() as u64)?;
        Ok(())
    }

    /// Perfrom a read from a memory-mapped IO area
    ///
    /// # Safety
    ///
    /// Caller must ensure that `pa` points to a properly aligned memory location and the
    /// memory accessed is part of a valid MMIO range.
    pub unsafe fn mmio_read(
        &self,
        pa: PhysAddr,
        value: &mut [MaybeUninit<u8>],
    ) -> Result<(), SvsmError> {
        self.clear();
        self.pepare_buffer();
        self.vmgexit(GHCBExitCode::MMIO_READ, u64::from(pa), value.len() as u64)?;
        self.read_buffer_slice(value, 0)?;
        Ok(())
    }

    pub fn psc_entry(
        &self,
        paddr: PhysAddr,
        op_mask: u64,
        current_page: u64,
        size: PageSize,
    ) -> u64 {
        assert!(size == PageSize::Regular || paddr.is_aligned(PAGE_SIZE_2M));

        let mut entry: u64 =
            ((paddr.bits() as u64) & PSC_GFN_MASK) | op_mask | (current_page & 0xfffu64);
        if size == PageSize::Huge {
            entry |= PSC_FLAG_HUGE;
        }

        entry
    }

    pub fn page_state_change(
        &self,
        region: MemoryRegion<PhysAddr>,
        op: PageStateChangeOp,
    ) -> Result<(), SvsmError> {
        // Maximum entries (8 bytes each_ minus 8 bytes for header
        let max_entries: u16 = ((GHCB_BUFFER_SIZE - 8) / 8).try_into().unwrap();
        let mut entries: u16 = 0;
        let mut paddr = region.start();
        let end = region.end();
        let op_mask: u64 = match op {
            PageStateChangeOp::Private => PSC_OP_PRIVATE,
            PageStateChangeOp::Shared => PSC_OP_SHARED,
            PageStateChangeOp::Psmash => PSC_OP_PSMASH,
            PageStateChangeOp::Unsmash => PSC_OP_UNSMASH,
        };

        self.clear();

        while paddr < end {
            let size = if paddr.is_aligned(PAGE_SIZE_2M) && paddr + PAGE_SIZE_2M <= end {
                PageSize::Huge
            } else {
                PageSize::Regular
            };
            let pgsize = usize::from(size);
            let entry = self.psc_entry(paddr, op_mask, 0, size);
            let offset = usize::from(entries) * 8 + 8;
            self.write_buffer(&entry, offset)?;
            entries += 1;
            paddr = paddr + pgsize;

            if entries == max_entries || paddr >= end {
                let header = PageStateChangeHeader {
                    cur_entry: 0,
                    end_entry: entries - 1,
                    reserved: 0,
                };
                self.write_buffer(&header, 0)?;

                self.pepare_buffer();

                if let Err(mut e) = self.vmgexit(GHCBExitCode::SNP_PSC, 0, 0) {
                    if let Err(err) = self.get_exit_info_2_valid() {
                        e = err;
                    }

                    if let GhcbError::VmgexitError(_, info2) = e {
                        let info_high: u32 = (info2 >> 32) as u32;
                        let info_low: u32 = (info2 & 0xffff_ffffu64) as u32;
                        log::error!(
                            "GHCB SnpPageStateChange failed err_high: {info_high:#x} err_low: {info_low:#x}"
                        );
                    }
                    return Err(e.into());
                }

                entries = 0;
            }
        }

        Ok(())
    }

    pub fn ap_create(
        &self,
        vmsa_gpa: PhysAddr,
        apic_id: u64,
        vmpl: u64,
        sev_features: u64,
    ) -> Result<(), SvsmError> {
        self.clear();
        let exit_info_1: u64 = 1 | (vmpl & 0xf) << 16 | apic_id << 32;
        let exit_info_2: u64 = vmsa_gpa.into();
        self.set_rax_valid(sev_features);
        self.vmgexit(GHCBExitCode::AP_CREATE, exit_info_1, exit_info_2)?;
        Ok(())
    }

    pub fn register_guest_vmsa(
        &self,
        vmsa_gpa: PhysAddr,
        apic_id: u64,
        vmpl: u64,
        sev_features: u64,
    ) -> Result<(), SvsmError> {
        self.clear();
        let exit_info_1: u64 = (vmpl & 0xf) << 16 | apic_id << 32;
        let exit_info_2: u64 = vmsa_gpa.into();
        self.set_rax_valid(sev_features);
        self.vmgexit(GHCBExitCode::AP_CREATE, exit_info_1, exit_info_2)?;
        Ok(())
    }

    pub fn register_hv_doorbell(&self, paddr: PhysAddr) -> Result<(), SvsmError> {
        self.clear();
        self.vmgexit(GHCBExitCode::HV_DOORBELL, 1, u64::from(paddr))?;
        Ok(())
    }

    pub fn guest_request(&self, req_page: VirtAddr, resp_page: VirtAddr) -> Result<(), SvsmError> {
        self.clear();

        let info1: u64 = u64::from(virt_to_phys(req_page));
        let info2: u64 = u64::from(virt_to_phys(resp_page));

        self.vmgexit(GHCBExitCode::GUEST_REQUEST, info1, info2)?;

        let sw_exit_info_2 = self.get_exit_info_2_valid()?;
        if sw_exit_info_2 != 0 {
            return Err(GhcbError::VmgexitError(
                self.sw_exit_info_1.load(Ordering::Relaxed),
                sw_exit_info_2,
            )
            .into());
        }

        Ok(())
    }

    pub fn guest_ext_request(
        &self,
        req_page: VirtAddr,
        resp_page: VirtAddr,
        data_pages: VirtAddr,
        data_size: u64,
    ) -> Result<(), SvsmError> {
        self.clear();

        let info1: u64 = u64::from(virt_to_phys(req_page));
        let info2: u64 = u64::from(virt_to_phys(resp_page));
        let rax: u64 = u64::from(virt_to_phys(data_pages));

        self.set_rax_valid(rax);
        self.set_rbx_valid(data_size);

        self.vmgexit(GHCBExitCode::GUEST_EXT_REQUEST, info1, info2)?;

        let sw_exit_info_2 = self.get_exit_info_2_valid()?;

        // On error, RBX and exit_info_2 are returned for proper error handling.
        // For an extended request, if the buffer provided is too small, the hypervisor
        // will return in RBX the number of contiguous pages required
        if sw_exit_info_2 != 0 {
            return Err(
                GhcbError::VmgexitError(self.rbx.load(Ordering::Relaxed), sw_exit_info_2).into(),
            );
        }

        Ok(())
    }

    pub fn hv_ipi(&self, icr: u64) -> Result<(), SvsmError> {
        self.clear();
        self.vmgexit(GHCBExitCode::HV_IPI, icr, 0)?;
        Ok(())
    }

    pub fn configure_interrupt_injection(&self, vector: usize) -> Result<(), SvsmError> {
        self.clear();
        self.vmgexit(GHCBExitCode::CONFIGURE_INT_INJ, vector as u64, 0)?;
        Ok(())
    }

    pub fn specific_eoi(&self, vector: u8, vmpl: u8) -> Result<(), SvsmError> {
        self.clear();
        let exit_info = ((vmpl as u64) << 16) | (vector as u64);
        self.vmgexit(GHCBExitCode::SPECIFIC_EOI, exit_info, 0)?;
        Ok(())
    }

    pub fn disable_alternate_injection(
        &self,
        tpr: u8,
        in_intr_shadow: bool,
        interrupts_enabled: bool,
    ) -> Result<(), SvsmError> {
        let mut exit_info = (GUEST_VMPL as u64) << 16;
        exit_info |= (tpr as u64) << 8;
        if in_intr_shadow {
            exit_info |= 2;
        }
        if interrupts_enabled {
            exit_info |= 1;
        }
        self.clear();
        self.vmgexit(GHCBExitCode::DISABLE_ALT_INJ, exit_info, 0)?;
        Ok(())
    }

    #[inline]
    #[cfg(test)]
    pub fn fill(&self, val: u8) {
        // SAFETY: All bytes in `Self` are part of an atomic integer type.
        // This allows us to cast `Self` to a slice of `AtomicU8`s.
        let bytes = unsafe {
            core::slice::from_raw_parts(self as *const _ as *const AtomicU8, size_of::<Self>())
        };
        for byte in bytes {
            byte.store(val, Ordering::Relaxed);
        }
    }
}

unsafe extern "C" {
    pub fn switch_to_vmpl_unsafe(hv_doorbell: *const HVDoorbell, vmpl: u32) -> bool;
}

pub fn switch_to_vmpl(vmpl: u32) {
    // The switch to a lower VMPL must be done with an assembly sequence in
    // order to ensure that any #HV that occurs during the sequence will
    // correctly block the VMPL switch so that events can be processed.
    let hv_doorbell = this_cpu().hv_doorbell();
    let ptr = match hv_doorbell {
        Some(doorbell) => ptr::from_ref(doorbell),
        None => ptr::null(),
    };
    // SAFETY: FFI call. Parameters and return values are checked.
    unsafe {
        if !switch_to_vmpl_unsafe(ptr, vmpl) {
            panic!("Failed to switch to VMPL {}", vmpl);
        }
    }
}

global_asm!(
    r#"
        .section .text
        .globl hv_redirect_window_start
    hv_redirect_window_start:

        .globl switch_to_vmpl_unsafe
    switch_to_vmpl_unsafe:

        /* Upon entry,
         * rdi = pointer to the HV doorbell page
         * esi = target VMPL
         */
        /* Check if NoFurtherSignal is set (bit 15 of the first word of the
         * #HV doorbell page).  If so, abort the transition. */
        test %rdi, %rdi
        jz switch_vmpl_proceed
        testw $0x8000, (%rdi)

        /* From this point until the vmgexit, if a #HV arrives, the #HV handler
         * must prevent the VMPL transition. */
        .globl switch_vmpl_window_start
    switch_vmpl_window_start:
        jnz switch_vmpl_cancel

    switch_vmpl_proceed:
        /* Use the MSR-based VMPL switch request to avoid any need to use the
         * GHCB page.  Run VMPL request is 0x16 and response is 0x17. */
        movl $0x16, %eax
        movl %esi, %edx
        movl ${SEV_GHCB}, %ecx
        wrmsr
        rep; vmmcall

        .globl switch_vmpl_window_end
    switch_vmpl_window_end:
        /* Verify that the request was honored.  ECX still contains the MSR
         * number. */
        rdmsr
        andl $0xFFF, %eax
        cmpl $0x17, %eax
        jz switch_vmpl_cancel
        xorl %eax, %eax
        ret

        /* An aborted VMPL switch is treated as a successful switch. */
        .globl switch_vmpl_cancel
    switch_vmpl_cancel:
        movl $1, %eax
        ret

        /* Idle halt is placed here as well, because idle halt requires
         * detection in the #HV handler to determine whether the HLT should be
         * skipped.  Placing this code here means that a single quick check can
         * be made to determine whether RIP is in a host exiting window (switch
         * VMPL or HLT) before needing to make a further determination about
         * which window was executing. */
        .globl snp_idle_halt
    snp_idle_halt:
        /* Upon entry, rdi = pointer to the HV doorbell page.
         * Check if NoFurtherSignal is set (bit 15 of the first word of the
         * #HV doorbell page).  If so, skip the halt. */
        test %rdi, %rdi
        jz idle_halt_proceed
        testw $0x8000, (%rdi)

        /* From this point until the HLT, if a #HV arrives, the #HV handler
         * must prevent the halt. */
        .globl idle_halt_window_start
    idle_halt_window_start:
        jnz idle_halt_window_end
    idle_halt_proceed:
        sti
        .globl snp_idle_hlt_instruction
    snp_idle_hlt_instruction:
        hlt
        .globl idle_halt_window_end
    idle_halt_window_end:
        cli
        ret

        .globl hv_redirect_window_end
    hv_redirect_window_end:

        "#,
    SEV_GHCB = const SEV_GHCB,
    options(att_syntax)
);

const _: () = assert!(
    offset_of!(GHCB, cpl) == 0x0cb
        && offset_of!(GHCB, xss) == 0x140
        && offset_of!(GHCB, dr7) == 0x160
        && offset_of!(GHCB, rax) == 0x1f8
        && offset_of!(GHCB, rcx) == 0x308
        && offset_of!(GHCB, rdx) == 0x310
        && offset_of!(GHCB, rbx) == 0x318
        && offset_of!(GHCB, rbp) == 0x328
        && offset_of!(GHCB, rsi) == 0x330
        && offset_of!(GHCB, rdi) == 0x338
        && offset_of!(GHCB, r8) == 0x340
        && offset_of!(GHCB, r9) == 0x348
        && offset_of!(GHCB, r10) == 0x350
        && offset_of!(GHCB, r11) == 0x358
        && offset_of!(GHCB, r12) == 0x360
        && offset_of!(GHCB, r13) == 0x368
        && offset_of!(GHCB, r14) == 0x370
        && offset_of!(GHCB, r15) == 0x378
        && offset_of!(GHCB, sw_exit_code) == 0x390
        && offset_of!(GHCB, sw_exit_info_1) == 0x398
        && offset_of!(GHCB, sw_exit_info_2) == 0x3a0
        && offset_of!(GHCB, sw_scratch) == 0x3a8
        && offset_of!(GHCB, xcr0) == 0x3e8
        && offset_of!(GHCB, valid_bitmap) == 0x3f0
        && offset_of!(GHCB, x87_state_gpa) == 0x400
        && offset_of!(GHCB, buffer) == 0x800
        && offset_of!(GHCB, version) == 0xffa
        && offset_of!(GHCB, usage) == 0xffc
        && mem::size_of::<GHCB>() == 0x1000
);

/// Test-only mock hypervisor for deterministic GHCB PSC reproduction.
///
/// This module injects at the VMGEXIT boundary only. Production
/// `GHCB::page_state_change()` / `GHCB::vmgexit()` response handling is not
/// rewritten; the mock mutates the live GHCB shared buffer and SW_EXITINFOx
/// fields in place, then returns into the unmodified guest-side checks.
#[cfg(test)]
pub mod psc_repro_mock {
    extern crate alloc;

    use super::*;
    use crate::locking::SpinLock;
    use alloc::vec::Vec;

    /// One hypervisor response applied on a single VMGEXIT.
    #[derive(Clone, Debug)]
    pub struct MockPscResponse {
        pub sw_exit_info_1: u64,
        pub sw_exit_info_2: u64,
        pub cur_entry: u16,
        pub end_entry: Option<u16>,
        pub reserved: Option<u32>,
        /// If set, overwrite entry[0].cur_page (low 12 bits).
        pub entry0_cur_page: Option<u16>,
    }

    impl MockPscResponse {
        pub fn ok_partial(cur_entry: u16) -> Self {
            Self {
                sw_exit_info_1: 0,
                sw_exit_info_2: 0,
                cur_entry,
                end_entry: None,
                reserved: None,
                entry0_cur_page: None,
            }
        }

        pub fn ok_complete(cur_entry: u16) -> Self {
            Self::ok_partial(cur_entry)
        }

        /// Linux `VMGEXIT_PSC_ERROR_INVALID_ENTRY` = ((1UL << 32) | 2).
        pub fn invalid_entry(cur_entry: u16) -> Self {
            Self {
                sw_exit_info_1: 0,
                sw_exit_info_2: 0x0000_0001_0000_0002,
                cur_entry,
                end_entry: None,
                reserved: None,
                entry0_cur_page: None,
            }
        }
    }

    struct MockState {
        responses: Vec<MockPscResponse>,
        next: usize,
        vmgexit_count: usize,
    }

    static MOCK: SpinLock<Option<MockState>> = SpinLock::new(None);
    static LAST_COUNT: SpinLock<usize> = SpinLock::new(0);

    pub fn install(responses: Vec<MockPscResponse>) {
        *MOCK.lock() = Some(MockState {
            responses,
            next: 0,
            vmgexit_count: 0,
        });
        *LAST_COUNT.lock() = 0;
    }

    pub fn uninstall() {
        if let Some(state) = MOCK.lock().take() {
            *LAST_COUNT.lock() = state.vmgexit_count;
        }
    }

    pub fn vmgexit_count() -> usize {
        MOCK.lock()
            .as_ref()
            .map(|s| s.vmgexit_count)
            .unwrap_or_else(|| *LAST_COUNT.lock())
    }

    /// Returns true if a mock handled this VMGEXIT.
    pub fn invoke_if_installed(ghcb: &GHCB) -> bool {
        let mut guard = MOCK.lock();
        let Some(state) = guard.as_mut() else {
            return false;
        };
        assert!(
            state.next < state.responses.len(),
            "mock hypervisor ran out of PSC responses (call #{})",
            state.vmgexit_count + 1
        );
        let resp = state.responses[state.next].clone();
        state.next += 1;
        state.vmgexit_count += 1;
        apply_response(ghcb, &resp);
        true
    }

    fn apply_response(ghcb: &GHCB, resp: &MockPscResponse) {
        // Overwrite SW_EXITINFO fields (validity bits already set by guest).
        ghcb.sw_exit_info_1
            .store(resp.sw_exit_info_1, Ordering::Relaxed);
        ghcb.sw_exit_info_2
            .store(resp.sw_exit_info_2, Ordering::Relaxed);

        let (_cur, mut end, mut reserved) = read_header(ghcb);
        let cur = resp.cur_entry;
        if let Some(e) = resp.end_entry {
            end = e;
        }
        if let Some(r) = resp.reserved {
            reserved = r;
        }
        write_header(ghcb, cur, end, reserved);

        if let Some(cur_page) = resp.entry0_cur_page {
            let mut entry = read_entry(ghcb, 0);
            entry = (entry & !0xfffu64) | (u64::from(cur_page) & 0xfff);
            write_entry(ghcb, 0, entry);
        }
    }

    pub fn read_header(ghcb: &GHCB) -> (u16, u16, u32) {
        let mut bytes = [0u8; 8];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ghcb.buffer[i].load(Ordering::Relaxed);
        }
        let cur = u16::from_le_bytes([bytes[0], bytes[1]]);
        let end = u16::from_le_bytes([bytes[2], bytes[3]]);
        let reserved = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        (cur, end, reserved)
    }

    fn write_header(ghcb: &GHCB, cur: u16, end: u16, reserved: u32) {
        let header = PageStateChangeHeader {
            cur_entry: cur,
            end_entry: end,
            reserved,
        };
        ghcb.write_buffer(&header, 0).expect("write PSC header");
    }

    pub fn read_entry(ghcb: &GHCB, index: usize) -> u64 {
        let offset = 8 + index * 8;
        let mut bytes = [0u8; 8];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ghcb.buffer[offset + i].load(Ordering::Relaxed);
        }
        u64::from_le_bytes(bytes)
    }

    fn write_entry(ghcb: &GHCB, index: usize, entry: u64) {
        let offset = 8 + index * 8;
        ghcb.write_buffer(&entry, offset)
            .expect("write PSC entry");
    }

    pub fn entry_cur_page(entry: u64) -> u16 {
        (entry & 0xfff) as u16
    }
}

/// Deterministic reproduction of GHCB PSC completion / error handling gaps.
///
/// These tests call the **upstream** `GHCB::page_state_change()` implementation
/// with a mock hypervisor injected only at the VMGEXIT boundary.
#[cfg(test)]
mod psc_repro_tests {
    extern crate alloc;
    // Host unit-test harness links std; use it only for diagnostic output.
    extern crate std;

    use super::psc_repro_mock::{self, MockPscResponse};
    use super::*;
    use crate::error::SvsmError;
    use crate::platform::PageStateChangeOp;
    use crate::types::{PAGE_SIZE, PAGE_SIZE_2M};
    use crate::utils::MemoryRegion;
    use alloc::boxed::Box;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;
    use std::eprintln;

    /// Allocate a zeroed GHCB page for userspace unit tests.
    fn new_test_ghcb() -> Box<GHCB> {
        Box::new(GHCB::new_zeroed())
    }

    fn region_4k(start: u64, pages: usize) -> MemoryRegion<PhysAddr> {
        MemoryRegion::new(PhysAddr::from(start), pages * PAGE_SIZE)
    }

    fn region_2m(start: u64) -> MemoryRegion<PhysAddr> {
        MemoryRegion::new(PhysAddr::from(start), PAGE_SIZE_2M)
    }

    fn format_result(result: &Result<(), SvsmError>) -> String {
        match result {
            Ok(()) => String::from("Ok(())"),
            Err(SvsmError::Ghcb(GhcbError::VmgexitError(i1, i2))) => {
                format!("Err(VmgexitError({i1:#x}, {i2:#x}))")
            }
            Err(e) => format!("Err({e:?})"),
        }
    }

    /// Test A: hypervisor only completes the first of three 4 KiB entries.
    ///
    /// Per GHCB §4.1.6 / Linux `vmgexit_psc()`, the guest must retry until
    /// `cur_entry > end_entry`. Upstream returns after a single VMGEXIT when
    /// SW_EXITINFO1 == 0.
    #[test]
    fn test_a_psc_partial_completion() {
        let ghcb = new_test_ghcb();
        psc_repro_mock::install(vec![
            MockPscResponse::ok_partial(1), // completed entry 0 only; end_entry still 2
            MockPscResponse::ok_complete(3), // cur_entry > end_entry => done
        ]);

        let region = region_4k(0x40_0000, 3);
        let result = ghcb.page_state_change(region, PageStateChangeOp::Private);
        let calls = psc_repro_mock::vmgexit_count();
        let (cur, end, _) = psc_repro_mock::read_header(&ghcb);
        psc_repro_mock::uninstall();

        eprintln!("=== Test A: psc_partial_completion ===");
        eprintln!("Expected VMGEXIT calls: 2");
        eprintln!("Actual VMGEXIT calls:   {calls}");
        eprintln!("Expected: continue until cur_entry > end_entry, then Ok");
        eprintln!(
            "Actual:   page_state_change returned {}, buffer cur_entry={cur} end_entry={end}",
            format_result(&result)
        );

        // BUG REPRODUCED if only one VMGEXIT and Ok(()).
        assert_eq!(
            calls, 1,
            "BUG REPRODUCED: expected 2 VMGEXIT calls for partial PSC, got {calls}"
        );
        assert!(
            result.is_ok(),
            "BUG REPRODUCED: upstream returned Ok after incomplete PSC"
        );
        assert!(
            cur <= end,
            "BUG REPRODUCED: buffer still incomplete (cur={cur} end={end}) but function returned Ok"
        );
        eprintln!("Result: BUG REPRODUCED");
    }

    /// Test B: PSC-specific error in SW_EXITINFO2 with SW_EXITINFO1 == 0.
    ///
    /// `VMGEXIT_PSC_ERROR_INVALID_ENTRY` = 0x00000001_00000002.
    /// Linux guest treats any non-zero SW_EXITINFO2 as failure.
    /// Upstream `vmgexit()` only checks SW_EXITINFO1.
    #[test]
    fn test_b_psc_error_in_exitinfo2() {
        let ghcb = new_test_ghcb();
        psc_repro_mock::install(vec![MockPscResponse::invalid_entry(1)]);

        let region = region_4k(0x40_0000, 3);
        let result = ghcb.page_state_change(region, PageStateChangeOp::Private);
        let calls = psc_repro_mock::vmgexit_count();
        let exit2 = ghcb.sw_exit_info_2.load(Ordering::Relaxed);
        psc_repro_mock::uninstall();

        eprintln!("=== Test B: psc_error_in_exitinfo2 ===");
        eprintln!("SW_EXITINFO1 = 0");
        eprintln!("SW_EXITINFO2 = {exit2:#018x}");
        eprintln!("Expected: Err(PSC invalid entry / non-zero EXITINFO2)");
        eprintln!("Actual:   {}", format_result(&result));
        eprintln!("VMGEXIT calls: {calls}");

        assert_eq!(calls, 1);
        assert_eq!(exit2, 0x0000_0001_0000_0002);
        assert!(
            result.is_ok(),
            "BUG REPRODUCED: expected PSC error, got {}",
            format_result(&result)
        );
        eprintln!("Result: BUG REPRODUCED");
    }

    /// Test C: 2 MiB entry partially completed (cur_page = 128 of 512).
    #[test]
    fn test_c_psc_2m_partial_cur_page() {
        let ghcb = new_test_ghcb();
        psc_repro_mock::install(vec![
            MockPscResponse {
                sw_exit_info_1: 0,
                sw_exit_info_2: 0,
                cur_entry: 0,
                end_entry: Some(0),
                reserved: None,
                entry0_cur_page: Some(128),
            },
            MockPscResponse {
                sw_exit_info_1: 0,
                sw_exit_info_2: 0,
                cur_entry: 1,
                end_entry: Some(0),
                reserved: None,
                entry0_cur_page: Some(512),
            },
        ]);

        // 2 MiB aligned GPA
        let region = region_2m(0x40_0000);
        let result = ghcb.page_state_change(region, PageStateChangeOp::Private);
        let calls = psc_repro_mock::vmgexit_count();
        let (cur, end, _) = psc_repro_mock::read_header(&ghcb);
        let entry0 = psc_repro_mock::read_entry(&ghcb, 0);
        let cur_page = psc_repro_mock::entry_cur_page(entry0);
        psc_repro_mock::uninstall();

        eprintln!("=== Test C: psc_2m_partial_cur_page ===");
        eprintln!("Expected VMGEXIT calls: 2 (resume from cur_page=128)");
        eprintln!("Actual VMGEXIT calls:   {calls}");
        eprintln!(
            "Buffer after return: cur_entry={cur} end_entry={end} entry0.cur_page={cur_page}"
        );
        eprintln!("Actual result: {}", format_result(&result));

        assert_eq!(
            calls, 1,
            "BUG REPRODUCED: expected retry for partial 2MiB PSC, got {calls} VMGEXIT(s)"
        );
        assert!(result.is_ok());
        assert_eq!(cur, 0);
        assert_eq!(end, 0);
        assert_eq!(cur_page, 128);
        eprintln!("Result: BUG REPRODUCED");
    }

    /// Test D: normal full completion in one VMGEXIT (negative control).
    #[test]
    fn test_d_psc_full_completion() {
        let ghcb = new_test_ghcb();
        psc_repro_mock::install(vec![MockPscResponse::ok_complete(3)]);

        let region = region_4k(0x40_0000, 3);
        let result = ghcb.page_state_change(region, PageStateChangeOp::Private);
        let calls = psc_repro_mock::vmgexit_count();
        let (cur, end, _) = psc_repro_mock::read_header(&ghcb);
        psc_repro_mock::uninstall();

        eprintln!("=== Test D: psc_full_completion ===");
        eprintln!("Expected VMGEXIT calls: 1");
        eprintln!("Actual VMGEXIT calls:   {calls}");
        eprintln!("Expected: Ok(())");
        eprintln!("Actual:   {}", format_result(&result));
        eprintln!("Buffer: cur_entry={cur} end_entry={end}");

        assert_eq!(calls, 1);
        assert!(result.is_ok());
        assert!(cur > end);
        eprintln!("Result: PASS (mock normal path OK)");
    }
}
