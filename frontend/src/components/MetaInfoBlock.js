import React from 'react';
import { Box, Typography, Table, TableBody, TableCell, TableRow, Paper, Chip, Tooltip } from '@mui/material';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import MemoryIcon from '@mui/icons-material/Memory';
import BugReportIcon from '@mui/icons-material/BugReport';
import InfoIcon from '@mui/icons-material/Info';

const formatMs = (ms) => ms != null ? `${ms} мс` : '—';
const formatTokens = (n) => n != null ? `${n} символов` : '—';
const formatDate = (iso) => iso ? new Date(iso).toLocaleString() : '—';
const formatSec = (ms) => ms != null ? `${(ms / 1000).toFixed(2)} сек` : '—';

const MetaInfoBlock = ({ meta }) => {
  return (
    <Paper elevation={1} sx={{ p: 2, mb: 2, bgcolor: '#f8fafc', borderRadius: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <InfoIcon color="primary" sx={{ mr: 1 }} />
        <Typography variant="subtitle2" color="text.secondary">
          Техническая информация о запросе
        </Typography>
      </Box>
      {!meta ? (
        <Typography variant="body2" color="text.disabled" sx={{ pl: 4 }}>
          Нет технической информации о последнем запросе
        </Typography>
      ) : (
        <Table size="small">
          <TableBody>
            <TableRow>
              <TableCell>Время поиска чанков</TableCell>
              <TableCell>
                <Chip icon={<AccessTimeIcon sx={{ fontSize: 16 }} />} label={formatSec(meta.retrieval_time_ms)} color={meta.retrieval_time_ms > 1000 ? 'warning' : 'success'} size="small" />
              </TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Время генерации ответа</TableCell>
              <TableCell>
                <Chip icon={<AccessTimeIcon sx={{ fontSize: 16 }} />} label={formatSec(meta.generation_time_ms)} color={meta.generation_time_ms > 3000 ? 'warning' : 'success'} size="small" />
              </TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Количество найденных чанков</TableCell>
              <TableCell>{meta.retrieved_chunks_count ?? '—'}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Длина prompt (символов)</TableCell>
              <TableCell>{formatTokens(meta.prompt_length)}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Длина ответа (символов)</TableCell>
              <TableCell>{formatTokens(meta.response_length)}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Модель</TableCell>
              <TableCell><Chip icon={<MemoryIcon sx={{ fontSize: 16 }} />} label={meta.model || '—'} size="small" /></TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Параметры генерации</TableCell>
              <TableCell>
                <Tooltip title="temperature">
                  <Chip label={`T=${meta.temperature ?? '—'}`} size="small" sx={{ mr: 1 }} />
                </Tooltip>
                <Tooltip title="top_k_chunks">
                  <Chip label={`top_k=${meta.top_k_chunks ?? '—'}`} size="small" />
                </Tooltip>
              </TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Trace ID</TableCell>
              <TableCell><Chip icon={<BugReportIcon sx={{ fontSize: 16 }} />} label={meta.trace_id || '—'} size="small" /></TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Время запроса</TableCell>
              <TableCell>{formatDate(meta.timestamp)}</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      )}
    </Paper>
  );
};

export default MetaInfoBlock;