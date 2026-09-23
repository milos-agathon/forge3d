use super::*;

pub(super) const SELECTION_READBACK_SLOTS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LodSelectionProvenance(pub(crate) u64);

#[derive(Debug)]
pub(crate) struct CompletedLodSelection {
    pub(crate) provenance: LodSelectionProvenance,
    pub(crate) selection: LodSelectionResult,
}

impl CompletedLodSelection {
    pub(crate) fn into_selection_for(
        self,
        expected: LodSelectionProvenance,
    ) -> Option<LodSelectionResult> {
        (self.provenance == expected).then_some(self.selection)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SelectionReadbackTicket {
    pub(super) id: u64,
    pub(super) slot: usize,
    pub(super) provenance: LodSelectionProvenance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SelectionReadbackTicketState {
    Staged,
    Submitted,
}

#[derive(Debug)]
pub(super) struct SelectionReadbackTickets {
    next_id: u64,
    slots:
        [Option<(SelectionReadbackTicket, SelectionReadbackTicketState)>; SELECTION_READBACK_SLOTS],
}

impl Default for SelectionReadbackTickets {
    fn default() -> Self {
        Self {
            next_id: 1,
            slots: [None; SELECTION_READBACK_SLOTS],
        }
    }
}

impl SelectionReadbackTickets {
    pub(super) fn stage(
        &mut self,
        provenance: LodSelectionProvenance,
    ) -> Option<SelectionReadbackTicket> {
        let slot = self.slots.iter().position(Option::is_none)?;
        let ticket = SelectionReadbackTicket {
            id: self.next_id,
            slot,
            provenance,
        };
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.slots[slot] = Some((ticket, SelectionReadbackTicketState::Staged));
        Some(ticket)
    }

    pub(super) fn mark_submitted(&mut self, ticket: SelectionReadbackTicket) -> bool {
        let Some((current, state)) = self.slots.get_mut(ticket.slot).and_then(Option::as_mut)
        else {
            return false;
        };
        if *current != ticket || *state != SelectionReadbackTicketState::Staged {
            return false;
        }
        *state = SelectionReadbackTicketState::Submitted;
        true
    }

    pub(super) fn cancel(&mut self, ticket: SelectionReadbackTicket) -> bool {
        if self.state(ticket) != Some(SelectionReadbackTicketState::Staged) {
            return false;
        }
        self.slots[ticket.slot] = None;
        true
    }

    pub(super) fn state(
        &self,
        ticket: SelectionReadbackTicket,
    ) -> Option<SelectionReadbackTicketState> {
        self.slots
            .get(ticket.slot)
            .and_then(|slot| *slot)
            .filter(|(current, _)| *current == ticket)
            .map(|(_, state)| state)
    }

    pub(super) fn oldest_submitted(&self) -> Option<SelectionReadbackTicket> {
        self.slots
            .iter()
            .flatten()
            .filter(|(_, state)| *state == SelectionReadbackTicketState::Submitted)
            .map(|(ticket, _)| *ticket)
            .min_by_key(|ticket| ticket.id)
    }

    pub(super) fn complete(&mut self, ticket: SelectionReadbackTicket) -> bool {
        if self.state(ticket) != Some(SelectionReadbackTicketState::Submitted) {
            return false;
        }
        self.slots[ticket.slot] = None;
        true
    }
}

pub(super) struct SelectionReadbackRuntime {
    tickets: SelectionReadbackTickets,
    receivers: [Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>;
        SELECTION_READBACK_SLOTS],
}

impl Default for SelectionReadbackRuntime {
    fn default() -> Self {
        Self {
            tickets: SelectionReadbackTickets::default(),
            receivers: [None, None],
        }
    }
}

/// GPU resources kept alive until the render pass consumes the indirect list.
pub struct GpuLodDrawResources {
    pub indirect_buffer: crate::core::resource_tracker::TrackedBuffer,
    pub instance_buffer: crate::core::resource_tracker::TrackedBuffer,
    pub output_tiles: crate::core::resource_tracker::TrackedBuffer,
    pub output_header: crate::core::resource_tracker::TrackedBuffer,
    pub max_draw_count: u32,
    pub(super) variant_count: u32,
    pub(super) params: crate::core::resource_tracker::TrackedBuffer,
    pub(super) input_tiles: crate::core::resource_tracker::TrackedBuffer,
    pub(super) _draw_templates: crate::core::resource_tracker::TrackedBuffer,
    pub(super) selection_readbacks:
        [crate::core::resource_tracker::TrackedBuffer; SELECTION_READBACK_SLOTS],
    pub(super) selection_readback: std::sync::Mutex<SelectionReadbackRuntime>,
    pub(super) bind_group: wgpu::BindGroup,
}

impl GpuLodDrawResources {
    fn selection_readback_size(&self) -> u64 {
        std::mem::size_of::<OutputHeader>() as u64
            + u64::from(self.max_draw_count) * std::mem::size_of::<TileInfo>() as u64
    }

    pub(super) fn stage_selection_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        provenance: LodSelectionProvenance,
    ) -> Option<SelectionReadbackTicket> {
        let ticket = self
            .selection_readback
            .lock()
            .expect("LOD selection readback mutex poisoned")
            .tickets
            .stage(provenance)?;
        let header_bytes = std::mem::size_of::<OutputHeader>() as u64;
        let tile_bytes = u64::from(self.max_draw_count) * std::mem::size_of::<TileInfo>() as u64;
        let readback = &self.selection_readbacks[ticket.slot];
        encoder.copy_buffer_to_buffer(&self.output_header, 0, readback, 0, header_bytes);
        encoder.copy_buffer_to_buffer(&self.output_tiles, 0, readback, header_bytes, tile_bytes);
        Some(ticket)
    }

    /// Start mapping the exact copy whose command buffer was submitted.
    pub(crate) fn mark_selection_submitted(&self, ticket: SelectionReadbackTicket) -> bool {
        let mut runtime = self
            .selection_readback
            .lock()
            .expect("LOD selection readback mutex poisoned");
        if !runtime.tickets.mark_submitted(ticket) {
            return false;
        }
        let (sender, receiver) = std::sync::mpsc::channel();
        self.selection_readbacks[ticket.slot].slice(..).map_async(
            wgpu::MapMode::Read,
            move |result| {
                let _ = sender.send(result);
            },
        );
        runtime.receivers[ticket.slot] = Some(receiver);
        true
    }

    /// Release a staged copy when its encoder is abandoned before submission.
    pub(crate) fn cancel_selection(&self, ticket: SelectionReadbackTicket) -> bool {
        let mut runtime = self
            .selection_readback
            .lock()
            .expect("LOD selection readback mutex poisoned");
        if runtime.receivers[ticket.slot].is_some() {
            return false;
        }
        runtime.tickets.cancel(ticket)
    }

    /// Poll once for a completed submitted selection. Pending tickets return
    /// `None`; this method never waits and never invents a CPU replacement.
    pub(crate) fn try_read_selection(
        &self,
        device: &wgpu::Device,
    ) -> RenderResult<Option<CompletedLodSelection>> {
        device.poll(wgpu::Maintain::Poll);
        let mut runtime = self
            .selection_readback
            .lock()
            .map_err(|_| RenderError::readback("LOD selection readback mutex poisoned"))?;
        let Some(ticket) = runtime.tickets.oldest_submitted() else {
            return Ok(None);
        };
        let slot = ticket.slot;
        let Some(receiver) = runtime.receivers[slot].as_ref() else {
            return Ok(None);
        };
        match receiver.try_recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                runtime.receivers[slot] = None;
                runtime.tickets.complete(ticket);
                return Err(RenderError::readback(format!(
                    "LOD selection map failed: {error}"
                )));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => return Ok(None),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                runtime.receivers[slot] = None;
                runtime.tickets.complete(ticket);
                return Err(RenderError::readback("LOD selection callback dropped"));
            }
        }

        let readback = &self.selection_readbacks[slot];
        let mapped = readback.slice(..).get_mapped_range();
        let result = self.parse_selection(&mapped);
        drop(mapped);
        readback.unmap();
        runtime.receivers[slot] = None;
        runtime.tickets.complete(ticket);
        result.map(|selection| {
            Some(CompletedLodSelection {
                provenance: ticket.provenance,
                selection,
            })
        })
    }

    fn parse_selection(&self, mapped: &[u8]) -> RenderResult<LodSelectionResult> {
        if mapped.len() != self.selection_readback_size() as usize {
            return Err(RenderError::readback("LOD selection readback size drift"));
        }
        let header_bytes = std::mem::size_of::<OutputHeader>();
        let header = bytemuck::pod_read_unaligned::<OutputHeader>(&mapped[..header_bytes]);
        let visible_count = header.visible_count.min(self.max_draw_count) as usize;
        let tile_size = std::mem::size_of::<TileInfo>();
        let visible_tiles = (0..visible_count)
            .map(|index| {
                let start = header_bytes + index * tile_size;
                bytemuck::pod_read_unaligned::<TileInfo>(&mapped[start..start + tile_size])
            })
            .collect();
        Ok(LodSelectionResult {
            visible_tiles,
            total_triangles: header.total_triangles,
            culled_count: self.max_draw_count - header.visible_count.min(self.max_draw_count),
        })
    }
}
